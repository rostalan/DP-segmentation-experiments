import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from ultralytics import YOLOWorld
from torchvision import models, transforms
from PIL import Image
import yaml
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config import load_config, merge_global_config

class ObjectSearcher:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        merge_global_config(self.config, Path(__file__).resolve().parent)
        
        model_path = self.config['paths'].get('yolo_world', '../models/yolov8x-worldv2.pt')
        self.conf_thresh = self.config['search'].get('confidence_threshold', 0.05)
        self.iou_thresh = self.config['search'].get('iou_threshold', 0.45)
        self.sim_thresh = self.config['search'].get('similarity_threshold', 0.60)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load YOLO-World for Generic Object Detection
        print(f"Loading YOLO-World from {model_path}...")
        self.yolo = YOLOWorld(model_path)
        self.yolo.to(self.device)
        
        # We use generic prompts to find potential candidates
        self.yolo.set_classes(["object", "thing", "item"])
        
        # Load Feature Extractor (ResNet18) for Re-ID
        print("Loading Feature Extractor (ResNet18)...")
        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor.fc = torch.nn.Identity()  # Remove classification layer
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Preprocessing for Feature Extractor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, img_crop):
        """Extract feature embedding from an image crop."""
        # Convert BGR (OpenCV) to RGB (PIL)
        if isinstance(img_crop, np.ndarray):
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_crop)
        else:
            img_pil = img_crop

        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.feature_extractor(img_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)  # Normalize
            
        return embedding

    def process_video(self):
        # 1. Load Reference Image(s)
        ref_path = self.config['paths']['reference_image']
        ref_images = []
        
        if isinstance(ref_path, list):
            for p in ref_path:
                img = cv2.imread(str(p))
                if img is None:
                    print(f"Warning: Could not read reference image: {p}")
                    continue
                ref_images.append(img)
        else:
            img = cv2.imread(str(ref_path))
            if img is None:
                raise ValueError(f"Could not read reference image: {ref_path}")
            ref_images.append(img)
            
        if not ref_images:
            raise ValueError("No valid reference images found.")

        print(f"Processing {len(ref_images)} reference image(s)...")
        
        # Compute embeddings for all reference images
        ref_embeddings = []
        for img in ref_images:
            emb = self.get_embedding(img)
            ref_embeddings.append(emb)
            
        # Stack embeddings: Shape (N_refs, Embedding_Dim)
        ref_embeddings = torch.cat(ref_embeddings, dim=0)

        # 2. Open Video
        video_path = self.config['paths']['input_video']
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {video_path} ({width}x{height} @ {fps:.2f}fps, {total_frames} frames)")
        
        # 3. Setup Output
        output_dir = Path(self.config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_writer = None
        if self.config['output'].get('save_video', True):
            video_name = self.config['output'].get('video_name', 'output_search.mp4')
            video_out_path = output_dir / video_name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
            print(f"Saving video to: {video_out_path}")

        save_frames = self.config['output'].get('save_frames', False)
        if save_frames:
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            
        # 4. Sampling Config
        start_frame = self.config['sampling'].get('start_frame', 0)
        frame_interval = self.config['sampling'].get('frame_interval', 1)
        max_frames = self.config['sampling'].get('max_frames')

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame_idx = start_frame + frame_count
            frame_count += 1
            
            if (frame_count - 1) % frame_interval != 0:
                if video_writer:
                    video_writer.write(frame)
                continue
            
            # Check max frames
            if max_frames and processed_count >= max_frames:
                break

            processed_count += 1
            
            # Process Frame
            # ---------------------------------------------------------
            results = self.yolo.predict(
                frame, 
                conf=self.conf_thresh, 
                iou=self.iou_thresh, 
                verbose=False
            )
            
            matches = []
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Boundary checks
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    crop = frame[y1:y2, x1:x2]
                    crop_embedding = self.get_embedding(crop)
                    
                    # Compute Cosine Similarity against ALL reference embeddings
                    # ref_embeddings: (N, D), crop_embedding: (1, D)
                    # Result: (N, 1) -> take max similarity
                    similarities = torch.mm(ref_embeddings, crop_embedding.t())
                    similarity = similarities.max().item()
                    
                    if similarity > self.sim_thresh:
                        matches.append((box, similarity))
            
            # Draw Matches
            annotated_frame = frame.copy()
            matches.sort(key=lambda x: x[1], reverse=True)
            
            for box, score in matches:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) # Green
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                label = f"Match {score:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Save Output
            if video_writer:
                video_writer.write(annotated_frame)
                
            if save_frames:
                frame_name = f"frame_{current_frame_idx:06d}.jpg"
                cv2.imwrite(str(frames_dir / frame_name), annotated_frame)

            elapsed = time.time() - start_time
            avg_fps = processed_count / elapsed if elapsed > 0 else 0
            print(f"Processed frame {current_frame_idx} | Matches: {len(matches)} | FPS: {avg_fps:.2f}")
            # ---------------------------------------------------------

        cap.release()
        if video_writer:
            video_writer.release()
            
        print(f"\nProcessing complete. Processed {processed_count} frames.")
        print(f"Output saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Search for an object in a video using YOLO-World + ResNet Re-ID")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    searcher = ObjectSearcher(config_path=args.config)
    searcher.process_video()

if __name__ == "__main__":
    main()
