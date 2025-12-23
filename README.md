# EveNet-Lite

EveNet-Lite is a minimal, PyTorch-first training helper that keeps the EveNet model stack but trims away heavy trainers or YAML-driven configuration. It exposes a small sklearn-like API (`fit/predict/evaluate`), a runner that wires up distributed training automatically, and convenience tools for checkpointing, sampling, normalization, and pretrained weight loading.

The repository is self contained; add the repo root to your `PYTHONPATH` or install in editable mode:

```bash
pip install -e .
```

## Quick start: pipeline runner

If you already have tensors prepared for objects/globals/mask, the runner wraps everything needed for a full train/validate/evaluate cycle and detects DDP from standard `torchrun` environment variables:

```python
import torch
from evenet_lite import run_evenet_lite_training

classifier = run_evenet_lite_training(
    train_features={"x": X_train, "globals": G_train, "mask": M_train},
    train_labels=y_train,
    train_weights=w_train,
    val_features={"x": X_val, "globals": G_val, "mask": M_val},
    val_labels=y_val,
    val_weights=w_val,
    eval_features={"x": X_test, "globals": G_test, "mask": M_test},
    eval_labels=y_test,
    eval_weights=w_test,
    class_labels=["background", "signal"],
    sampler="weighted",
    epochs=5,
    batch_size=512,
    checkpoint_path="./checkpoints",
    save_top_k=2,
    debug=True,
)

# The returned classifier is already fitted and carries the trained normalizer.
probs = classifier.predict({"x": X_infer, "globals": G_infer, "mask": M_infer})
metrics = classifier.evaluate({"x": X_eval, "globals": G_eval, "mask": M_eval}, y_eval)
classifier.save_checkpoint("./checkpoints/final.pt")
```

## Parameterized training (m_X, m_Y) example

To train with per-event parameters (e.g., ``m_X`` and ``m_Y``) and randomize background
values every step, add a ``params`` tensor to your feature dictionaries and attach a
``ParameterRandomizationCallback``. The helper script in ``examples/parameterized_train_multi_gpu_example.py``
mirrors ``train_multi_gpu.py``'s concatenation style and runs a tiny synthetic job:

```bash
python examples/parameterized_train_multi_gpu_example.py
```

Key steps to adapt the pattern to your own ``train_multi_gpu.py`` workflow:

1. Save ``params`` alongside ``x``, ``x_mask``, and ``global`` in your ``.pt`` tensors; the loader now
   automatically includes it when present.
2. Increase ``--global-input-dim`` (e.g., ``10 + number_of_params``) so the model expects the expanded global size.
3. Pass ``ParameterRandomizationCallback`` in your callbacks list to resample background parameters each batch
   while leaving signal parameters intact.

### What the runner handles

- Detects distributed environments (`WORLD_SIZE`, `LOCAL_RANK`) and pins the appropriate CUDA device when available.
- Builds an `EvenetLiteClassifier` with optional pretrained weights and logging level.
- Injects normalization automatically unless a custom `NormalizationCallback` is supplied.
- Forwards checkpointing, early stopping, sampler, and evaluation options to the classifier.
- Returns the fitted classifier so you can immediately call `predict`, `evaluate`, or `save_checkpoint`.

## Custom workflow (manual steps)

Prefer to assemble the pieces yourself? You can directly instantiate the classifier, call `fit`, and run evaluation/prediction without the runner.

```python
import torch
from evenet_lite import EvenetLiteClassifier

# Build the classifier (uses default EveNetLite backbone if none is provided)
clf = EvenetLiteClassifier(
    class_labels=["background", "signal"],
    device="auto",          # cpu, cuda, or auto-detect
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=1.0,
    pretrained=True,         # load HF weights by default
)

# Fit
clf.fit(
    train_data=({"x": X_train, "globals": G_train, "mask": M_train}, y_train, w_train),
    val_data=({"x": X_val, "globals": G_val, "mask": M_val}, y_val, w_val),
    feature_names={"x": obj_feature_names, "globals": global_feature_names},
    epochs=10,
    batch_size=256,
    sampler="weighted",
    checkpoint_path="./checkpoints",
    save_top_k=1,
)

# Evaluate / predict
val_metrics = clf.evaluate({"x": X_val, "globals": G_val, "mask": M_val}, y_val, w_val)
probs = clf.predict({"x": X_test, "globals": G_test, "mask": M_test})

# Checkpointing
clf.save_checkpoint("./checkpoints/latest.pt")
# Later
restored = EvenetLiteClassifier(class_labels=["background", "signal"])
restored.load_checkpoint("./checkpoints/latest.pt", feature_names={"x": obj_feature_names, "globals": global_feature_names})
```

## Data expectations

Input tensors follow an xgboost-like contract and are provided directly to the classifier or runner:

```python
features = {
    "x": torch.Tensor[N, M, F],     # per-object features
    "globals": torch.Tensor[N, G],  # event-level features
    "mask": torch.Tensor[N, M],     # padding mask
}
labels = torch.Tensor[N]              # class indices
weights = torch.Tensor[N] | None      # optional per-example weights
```

Feature names passed to `fit` (`feature_names={"x": [...], "globals": [...]}`) should align with the keys above so the normalizer can match statistics to columns.

## Argument reference

The tables below summarize the most-used entrypoints and their arguments. Defaults match the inline values in code.

### `EvenetLiteClassifier` constructor

| Argument | Default | Description |
| --- | --- | --- |
| `class_labels` | **required** | Ordered class names wired into metrics and loss. |
| `device` | `"auto"` | Chooses CUDA when available; otherwise CPU. |
| `lr` | `DEFAULT_HEAD_LR` | Base learning rate for the head (and body if `body_lr` not set). |
| `weight_decay` | `DEFAULT_WEIGHT_DECAY` | Global weight decay when per-group values are not provided. |
| `model` | `None` | Custom EveNet model; defaults to `EveNetLite` built from `config/default_network_config.yaml`. |
| `optimizer_fn` / `scheduler_fn` | `None` | Factories for custom optimizer or scheduler. |
| `grad_clip` | `None` | Max gradient norm when set. |
| `body_lr` / `head_lr` | `None` | Overrides for body/head learning rates (body defaults to `0.1 * head`). |
| `body_weight_decay` / `head_weight_decay` | `None` | Overrides for weight decay by parameter group. |
| `body_modules` / `head_modules` | `DEFAULT_*_MODULES` | Module name prefixes assigned to body/head parameter groups. |
| `warmup_epochs` / `warmup_ratio` / `warmup_start_factor` | `1` / `0.1` / `0.1` | Linear warmup configuration. |
| `min_lr` | `0.0` | Scheduler floor learning rate. |
| `global_input_dim` / `sequential_input_dim` | `10` / `7` | Input feature dimensions for the default backbone. |
| `use_wandb` / `wandb` | `False` / `None` | Enable Weights & Biases with optional init kwargs. |
| `log_level` | `logging.INFO` | Root logging level when constructing the classifier. |
| `pretrained` | `False` | When `True`, soft-loads weights (default HF repo/filename). |
| `pretrained_source` | `"hf"` | `"hf"` for Hugging Face hub or `"local"` for a provided path. |
| `pretrained_path` / `pretrained_repo_id` / `pretrained_filename` / `pretrained_cache_dir` | varies | Location details for pretrained checkpoints. |

### `EvenetLiteClassifier.fit`

| Argument | Default | Description |
| --- | --- | --- |
| `train_data` | **required** | Tuple `(features, labels, weights)` for training. |
| `val_data` | `None` | Optional validation tuple with same structure as training. |
| `feature_names` | Defaults to classifier presets | Mapping of feature group to column names for normalization. |
| `normalization_rules` | Defaults to classifier presets | Per-feature normalization strategy (`log_normalize`, `normalize`, etc.). |
| `callbacks` | `None` | Additional callbacks (normalization is auto-inserted if absent). |
| `epochs` | `10` | Number of training epochs. |
| `batch_size` | `256` | Mini-batch size. |
| `sampler` | `None` | Sampler name (`"weighted"` enables distributed-safe weighted sampler). |
| `epoch_size` | `None` | Number of samples per epoch when using a sampler. |
| `checkpoint_path` / `resume_from` | `None` | Directory or filename for checkpoints and optional resume path. |
| `checkpoint_every` | `1` | Frequency (epochs) for periodic checkpoints when `save_top_k == 0`. |
| `save_top_k` | `0` | Keep best-k checkpoints ranked by `monitor_metric`. |
| `monitor_metric` / `minimize_metric` | `"val_loss"` / `True` | Metric and direction for checkpoint ranking. |
| `early_stop_metric` / `early_stop_minimize` / `early_stop_patience` | `"val_loss"` / `True` / `0` | Early stopping configuration (disabled when patience is 0). |
| `eval_data` | `None` | Optional test tuple evaluated after training. |
| `eval_output_path` | `None` | Path to save evaluation outputs when provided. |
| `eval_batch_size` | `None` | Batch size for evaluation (falls back to training batch size). |
| `sic_min_bkg_events` | `100` | Minimum background events for SIC metric calculation. |
| `debug` | `False` | Enables verbose `DebugCallback` logging and diagnostics. |

### `EvenetLiteClassifier.predict` / `evaluate`

- `predict(features, batch_size=256)`: returns class probabilities using the stored normalizer; requires that `fit` or `load_checkpoint` has been called.
- `evaluate(features, labels, weights=None, batch_size=256)`: computes loss/accuracy (and physics metrics when available) on the provided dataset.

### `run_evenet_lite_training`

| Argument | Default | Description |
| --- | --- | --- |
| `train_features` / `train_labels` / `train_weights` | **required** / **required** / `None` | Training tensors and optional weights. |
| `class_labels` | **required** | Ordered class names passed to the classifier. |
| `val_features` / `val_labels` / `val_weights` | `None` | Optional validation tensors and weights. |
| `feature_names` | `None` | Feature column names forwarded to the classifier. |
| `normalization_rules` | `None` | Per-feature normalization overrides. |
| `callbacks` | `None` | Extra callbacks (normalization auto-added if missing). |
| `sampler` / `epoch_size` | `None` | Sampling strategy and epoch size when sampling. |
| `epochs` / `batch_size` | `10` / `256` | Training loop configuration. |
| `checkpoint_path` / `resume_from` | `None` | Checkpoint directory/base filename and optional resume path. |
| `checkpoint_every` | `1` | Epoch frequency for periodic checkpoints when not using top-k. |
| `save_top_k` | `0` | Number of best checkpoints to retain. |
| `monitor_metric` / `minimize_metric` | `"val_loss"` / `True` | Metric and direction for best-checkpoint tracking. |
| `early_stop_metric` / `early_stop_minimize` / `early_stop_patience` | `"val_loss"` / `True` / `0` | Early stopping configuration. |
| `eval_features` / `eval_labels` / `eval_weights` | `None` | Optional evaluation payload run after training. |
| `eval_output_path` | `None` | File path to persist evaluation results. |
| `eval_batch_size` | `None` | Batch size for evaluation (defaults to training batch size). |
| `sic_min_bkg_events` | `100` | Minimum background events for SIC metric computation. |
| `debug` | `False` | Enables verbose debugging callback and sampler diagnostics. |
| `log_level` | `logging.INFO` | Logging level set before runner diagnostics. |
| `**classifier_kwargs` | — | Additional arguments forwarded directly to `EvenetLiteClassifier`. |

## Distributed training

The trainer boots into DDP automatically when `WORLD_SIZE > 1` (e.g., via `torchrun --nproc_per_node <num_gpus> script.py`). Rank 0 handles logging and checkpointing; sampler/loader seeds are synchronized per epoch. Without distributed environment variables, execution falls back to single process on GPU or CPU depending on availability.

## Normalization & callbacks

- A `NormalizationCallback` is injected automatically during `fit` when one is not provided. You can supply custom normalization rules or replace the callback entirely.
- Implement custom callbacks by subclassing `Callback` and overriding hooks such as `on_train_start`, `on_epoch_end`, or `on_train_end`, then pass instances via the `callbacks` argument of `fit` or the runner.

## Checkpointing and pretrained weights

- Call `save_checkpoint(path)` on a fitted classifier to persist model, optimizer/scheduler states, and the learned normalizer. Use `load_checkpoint(path, feature_names=...)` to restore weights for further training or inference.
- Enable `pretrained=True` (with optional `pretrained_source`, `pretrained_path`, or Hugging Face repo/filename overrides) to soft-load compatible parameters while leaving shape-mismatched layers initialized.

## Module guide

- `evenet_lite.classifier.EvenetLiteClassifier`: high-level `fit/predict/evaluate` API and pretrained loader.
- `evenet_lite.runner.run_evenet_lite_training`: convenience pipeline that wires up DDP detection and training.
- `evenet_lite.trainer.Trainer`: core training loop with DDP, callbacks, metrics, early stopping, and checkpointing.
- `evenet_lite.data`: dataset wrapper and distributed weighted sampler utilities.
- `evenet_lite.callbacks`: callback base class, default normalizer, and debug helpers.
- `evenet_lite.metrics`: accuracy, loss, and physics-driven metrics helpers.
- `evenet_lite.checkpoint`: rank-safe checkpoint save/load helpers.
- `evenet_lite.model`: EveNet backbone assembly used by the default classifier.
