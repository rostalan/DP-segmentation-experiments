"""Shared YAML config loading with global_config.yaml merging."""

import yaml
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _apply_defaults(target: dict, defaults: dict) -> None:
    """Add keys from *defaults* into *target* only where missing.

    For nested dicts the operation recurses so that local overrides
    at any depth are preserved.
    """
    for key, val in defaults.items():
        if key not in target:
            target[key] = val
        elif isinstance(target[key], dict) and isinstance(val, dict):
            _apply_defaults(target[key], val)


def merge_global_config(config: dict, script_dir: Path) -> dict:
    """Merge ``global_config.yaml`` defaults into the local *config*.

    Global values are applied only when the local config does not already
    define the key (at any nesting depth).  Model paths land under
    ``config["paths"]``.  Returns *config* for convenience.
    """
    global_path = (script_dir / ".." / "global_config.yaml").resolve()
    if not global_path.exists():
        return config

    with open(global_path, "r") as f:
        gc = yaml.safe_load(f)

    if "colors" in gc:
        config["_global_colors"] = gc.pop("colors")

    if "models" in gc:
        models = gc.pop("models")
        config.setdefault("paths", {})
        for k, v in models.items():
            config["paths"].setdefault(k, v)

    _apply_defaults(config, gc)
    return config


def get_colors(config: dict) -> list[list[int]]:
    """Return the colour palette from merged config or a sensible default."""
    return config.get("colors", config.get("_global_colors", [
        [75, 25, 230], [75, 180, 60], [25, 225, 255], [200, 130, 0],
        [48, 130, 245], [180, 30, 145], [240, 240, 70], [230, 50, 240],
        [60, 245, 210], [212, 190, 250], [128, 128, 0], [255, 190, 220],
    ]))
