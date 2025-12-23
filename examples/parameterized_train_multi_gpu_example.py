"""Minimal example showing parameterized training with train_multi_gpu-style data.

This script constructs toy signal/background tensors with two physics parameters
(m_X, m_Y), concatenates them into the ``params`` feature, and trains for a few
steps while dynamically randomizing background parameters via
``ParameterRandomizationCallback``.
"""

from __future__ import annotations

import torch

from evenet_lite import run_evenet_lite_training
from evenet_lite.callbacks import ParameterRandomizationCallback


def _build_split(num_events: int, m_x: float, m_y: float, label: int) -> tuple[dict, torch.Tensor]:
    """Create a tiny synthetic dataset with globals, params, and sequence data."""

    num_objects = 6
    sequential_dim = 7
    global_dim = 10

    x = torch.rand(num_events, num_objects, sequential_dim)
    x_mask = torch.ones(num_events, num_objects)
    globals_tensor = torch.rand(num_events, global_dim)
    params = torch.stack(
        [torch.full((num_events,), m_x, dtype=torch.float32), torch.full((num_events,), m_y, dtype=torch.float32)],
        dim=1,
    )

    return {"x": x, "x_mask": x_mask, "globals": globals_tensor, "params": params}, torch.full(
        (num_events,), label, dtype=torch.long
    )


def main() -> None:
    # Build simple signal/background slices with distinct (m_X, m_Y)
    sig_features, sig_labels = _build_split(num_events=256, m_x=1200.0, m_y=90.0, label=1)
    bkg_features, bkg_labels = _build_split(num_events=256, m_x=900.0, m_y=140.0, label=0)

    # Merge along the batch dimension to mirror train_multi_gpu.py's concatenation logic
    train_features = {key: torch.cat([sig_features[key], bkg_features[key]], dim=0) for key in sig_features}
    train_labels = torch.cat([sig_labels, bkg_labels], dim=0)

    # Optional validation split reuses the same construction for brevity
    val_features = {key: value.clone() for key, value in train_features.items()}
    val_labels = train_labels.clone()

    # Use the callback to randomize background parameters each step inside the provided bounds
    param_callback = ParameterRandomizationCallback(
        param_key="params",
        background_label=0,
        min_values=[600.0, 40.0],
        max_values=[1500.0, 250.0],
    )

    run_evenet_lite_training(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        class_labels=["background", "signal"],
        callbacks=[param_callback],
        # Globals are 10-D and we append 2 parameters (m_X, m_Y)
        global_input_dim=12,
        epochs=1,
        batch_size=64,
        debug=True,
    )


if __name__ == "__main__":
    main()
