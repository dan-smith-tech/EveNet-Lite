

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    normalizer_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model": model_state,
        "optimizer": optimizer_state,
        "normalizer": normalizer_state or {},
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
