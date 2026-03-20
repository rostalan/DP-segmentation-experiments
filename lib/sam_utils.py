"""Shared SAM 3 interactive selector and custom video predictor."""

from __future__ import annotations
import cv2
import numpy as np


class InteractiveSelector:
    """Interactive bounding-box selector on a frame (for SAM 3 init)."""

    def __init__(self, frame: np.ndarray,
                 window_name: str = "Select Objects",
                 scale_factor: float = 2 / 3):
        self.original_frame = frame.copy()
        self.scale_factor = scale_factor
        h, w = frame.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        self.display_frame = cv2.resize(frame, (new_w, new_h))
        self.current_frame = self.display_frame.copy()
        self.window_name = window_name
        self.boxes: list[list[float]] = []
        self.current_box_start = None
        self.drawing = False

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_frame = self.display_frame.copy()
            for box in self.boxes:
                x1, y1, x2, y2 = [c * self.scale_factor for c in box]
                cv2.rectangle(self.current_frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
            cv2.rectangle(self.current_frame, self.current_box_start,
                          (x, y), (0, 0, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            xs, ys = self.current_box_start
            x1d, x2d = sorted([xs, x])
            y1d, y2d = sorted([ys, y])
            if (x2d - x1d) > 5 and (y2d - y1d) > 5:
                self.boxes.append([
                    x1d / self.scale_factor, y1d / self.scale_factor,
                    x2d / self.scale_factor, y2d / self.scale_factor,
                ])
                print(f"Added box: {self.boxes[-1]}")
            self._redraw()

    def _redraw(self):
        self.current_frame = self.display_frame.copy()
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = [c * self.scale_factor for c in box]
            cv2.rectangle(self.current_frame,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(self.current_frame, f"ID: {i}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(self.current_frame,
                    "Draw boxes. SPACE/ENTER to finish, 'r' to reset, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    def select(self) -> list[list[float]]:
        self._redraw()
        print("Draw bounding boxes. SPACE/ENTER to confirm, "
              "'r' to clear, 'q' to quit.")
        while True:
            cv2.imshow(self.window_name, self.current_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord(" "), 13):
                if not self.boxes:
                    print("No boxes selected.")
                    continue
                break
            elif key == ord("r"):
                self.boxes = []
                self._redraw()
                print("Selection cleared.")
            elif key == ord("q"):
                cv2.destroyWindow(self.window_name)
                return []
        cv2.destroyWindow(self.window_name)
        return self.boxes


def make_custom_sam3_predictor():
    """Return a CustomSAM3VideoPredictor class (import-guarded).

    The SAM3VideoPredictor base lives in ultralytics and may not be installed,
    so we defer the import and subclass creation.
    """
    import torch
    from ultralytics.models.sam import SAM3VideoPredictor

    class CustomSAM3VideoPredictor(SAM3VideoPredictor):
        """Prevents filtering of empty masks to ensure persistent object IDs."""

        def inference(self, im, bboxes=None, points=None, labels=None,
                      masks=None):
            bboxes = self.prompts.pop("bboxes", bboxes)
            points = self.prompts.pop("points", points)
            masks = self.prompts.pop("masks", masks)

            frame = self.dataset.frame
            self.inference_state["im"] = im
            output_dict = self.inference_state["output_dict"]

            if len(output_dict["cond_frame_outputs"]) == 0:
                points, labels, masks = self._prepare_prompts(
                    im.shape[2:], self.batch[1][0].shape[:2],
                    bboxes, points, labels, masks,
                )
                if points is not None:
                    for i in range(len(points)):
                        self.add_new_prompts(
                            obj_id=i, points=points[[i]],
                            labels=labels[[i]], frame_idx=frame)
                elif masks is not None:
                    for i in range(len(masks)):
                        self.add_new_prompts(
                            obj_id=i, masks=masks[[i]], frame_idx=frame)

            self.propagate_in_video_preflight()

            consolidated = self.inference_state["consolidated_frame_inds"]
            batch_size = len(self.inference_state["obj_idx_to_id"])

            if len(output_dict["cond_frame_outputs"]) == 0:
                raise RuntimeError("No points provided; add points first")

            if frame in consolidated["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame]
                if (self.clear_non_cond_mem_around_input
                        and (self.clear_non_cond_mem_for_multi_obj
                             or batch_size <= 1)):
                    self._clear_non_cond_mem_around_input(frame)
            elif frame in consolidated["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out = self._run_single_frame_inference(
                    output_dict=output_dict, frame_idx=frame,
                    batch_size=batch_size, is_init_cond_frame=False,
                    point_inputs=None, mask_inputs=None,
                    reverse=False, run_mem_encoder=True,
                )
                output_dict[storage_key][frame] = current_out
                self._prune_non_cond_memory(frame)

            self._add_output_per_object(frame, current_out, storage_key)
            self.inference_state["frames_already_tracked"].append(frame)

            pred_masks = current_out["pred_masks"].flatten(0, 1)
            return pred_masks, torch.ones(
                pred_masks.shape[0],
                dtype=pred_masks.dtype,
                device=pred_masks.device,
            )

    return CustomSAM3VideoPredictor
