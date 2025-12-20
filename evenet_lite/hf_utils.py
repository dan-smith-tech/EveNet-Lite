from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache


def load_pretrained_weights(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> Optional[dict]:
    """Load pretrained weights from the Hugging Face Hub.

    Environment variables respected:
        - ``EVENET_MODEL_PATH``: if set, loads directly from this path.
        - ``HF_TOKEN``: authentication token for private repos.
    """

    env_path = os.getenv("EVENET_MODEL_PATH")
    if env_path:
        resolved = Path(env_path).expanduser()
        if resolved.is_file():
            return torch.load(resolved, map_location="cpu")

    token = os.getenv("HF_TOKEN")
    cache_path = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=None)
    if cache_path is None:
        cache_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=token)
    if cache_path and Path(cache_path).exists():
        return torch.load(cache_path, map_location="cpu")
    return None
