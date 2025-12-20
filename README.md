# Evenet-Lite Framework

Evenet-Lite is a minimal, PyTorch-first training helper designed around an sklearn-style API for training classification heads on top of an EveNet core model. The framework exposes a small set of composable utilities for training, evaluation, checkpointing, sampling, and normalization without relying on PyTorch Lightning or YAML configuration.

## Installation

The code is self contained inside this repository. Add the repo root to your `PYTHONPATH` or install in editable mode:

```bash
pip install -e .
```

## Core API

### High-level classifier

```python
from evenet_lite import EvenetLiteClassifier

clf = EvenetLiteClassifier(
    model=evenet_core_model,
    num_classes=2,
    device="auto",        # "cpu", "cuda", or "auto"
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=1.0,         # optional gradient clipping
)

clf.fit(
    train_data=(X_train, y_train, w_train),
    val_data=(X_val, y_val, w_val),
    feature_names={"objects": obj_feature_names, "globals": global_feature_names},
    callbacks=[],           # custom callbacks optional; normalization auto-injected
    epochs=10,
    batch_size=256,
    sampler="weighted",    # or None
    epoch_size=None,
)

probs = clf.predict(X_test, batch_size=256)
metrics = clf.evaluate(X_test, y_test, w_test, batch_size=256)
```

### Data expectations

Input tensors follow an xgboost-like contract and are provided directly rather than read from disk:

```python
X = {
    "objects": torch.Tensor[N, M, F],
    "globals": torch.Tensor[N, G],
    "mask": torch.Tensor[N, M],
}
y = torch.Tensor[N]                # class indices
weights = torch.Tensor[N] | None   # optional per-example weights
```

Feature names (`feature_names`) should mirror the keys in `X` so the normalizer knows which tensors to normalize.

### Normalization

Normalization is handled through callbacks. A default `NormalizationCallback` is automatically attached during `fit`, which uses an `EvenetLiteNormalizer` to compute statistics on the training set and reuse them for validation/testing. You can provide your own normalizer callback to override the behavior.

### Custom callbacks

Subclass `Callback` and override any of the hook methods (`on_train_start`, `on_epoch_start`, `on_batch_end`, `on_epoch_end`, `on_train_end`) to inject custom logic. Pass instances via the `callbacks` argument of `fit`.

### Samplers and class imbalance

Set `sampler="weighted"` in `fit` to enable the distributed-safe weighted sampler. Provide `weights` in your training tuple to use custom example weights; otherwise, class weights are derived from label frequencies. Use `epoch_size` to control how many samples each epoch draws when the sampler is active.

### Distributed training

The trainer automatically detects `torch.distributed` initialization and wraps the model with `DistributedDataParallel`. Launch multi-GPU training with `torchrun` or `torch.distributed.launch`; logging and checkpointing occur only on rank 0.

### Checkpointing

Use `save_checkpoint` and `load_checkpoint` to persist or restore model, optimizer, scheduler, and callback state. These helpers are rank-safe and work with both single-GPU and DDP runs.

### Hugging Face weight loading

`load_pretrained_weights` downloads weights from the Hugging Face Hub while respecting the `EVENET_MODEL_PATH` and `HF_TOKEN` environment variables. The loader is shape-safe and allows partial weight loading so you can warm-start compatible parameters.

## Module overview

- `evenet_lite.classifier.EvenetLiteClassifier`: sklearn-like `fit`, `predict`, `evaluate` entrypoint.
- `evenet_lite.trainer.Trainer`: core training loop with DDP, callbacks, metrics, and prediction utilities.
- `evenet_lite.callbacks`: callback base class, default normalizer, and normalization callback.
- `evenet_lite.data`: tensor dataset wrapper and distributed weighted sampler for imbalance handling.
- `evenet_lite.metrics`: built-in accuracy and AUC helpers.
- `evenet_lite.checkpoint`: checkpoint save/load helpers.
- `evenet_lite.hf_utils`: Hugging Face Hub download helper.

## Minimal end-to-end example

```python
from evenet_lite import EvenetLiteClassifier

# example EveNet core model producing logits
core_model = build_eve_net_core(num_classes=2)

trainer = EvenetLiteClassifier(core_model, num_classes=2, device="auto")
trainer.fit(
    train_data=(X_train, y_train, None),
    val_data=(X_val, y_val, None),
    feature_names={"objects": obj_feature_names, "globals": global_feature_names},
    epochs=5,
    batch_size=512,
    sampler="weighted",
)
print(trainer.evaluate(X_test, y_test))
```
