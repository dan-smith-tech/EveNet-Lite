import logging
from typing import Dict, Optional

import torch

from . import Callback


class DebugCallback(Callback):
    """Verbose, rank-safe logging helper for rapid debugging.

    Logs sampler summaries at the start of each epoch, gradient norms after
    backward passes, and per-batch/epoch metrics. Only the global rank 0
    process emits logs when running under DDP.
    """

    def __init__(self, log_every_n_batches: int = 10) -> None:
        self.log_every_n_batches = max(1, log_every_n_batches)

    def _grad_norms(self, model: torch.nn.Module) -> Optional[Dict[str, float]]:
        norms = [p.grad.detach().data.norm(2) for p in model.parameters() if p.grad is not None]
        if not norms:
            return None
        stacked = torch.stack(norms)
        total = torch.sqrt(torch.sum(stacked**2)).item()
        return {"total": float(total), "max": float(stacked.max().item())}

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        if not trainer.is_rank_zero():
            return
        train_sampler = trainer.train_sampler if hasattr(trainer, "train_sampler") else None
        val_sampler = trainer.val_sampler if hasattr(trainer, "val_sampler") else None
        train_loader = trainer.train_loader if hasattr(trainer, "train_loader") else None
        val_loader = trainer.val_loader if hasattr(trainer, "val_loader") else None

        train_sampler_desc = trainer._describe_sampler(train_sampler) if train_sampler is not None else "None"
        val_sampler_desc = trainer._describe_sampler(val_sampler) if val_sampler is not None else "None"
        train_steps = len(train_loader) if train_loader is not None else 0
        val_steps = len(val_loader) if val_loader is not None else 0

        logging.info(
            "[Debug] Epoch %d start | train_steps=%d | train_sampler=%s | world_size=%d",
            epoch + 1,
            train_steps,
            train_sampler_desc,
            trainer.world_size,
        )
        if val_loader is not None:
            logging.info(
                "[Debug] Epoch %d validation | val_steps=%d | val_sampler=%s",
                epoch + 1,
                val_steps,
                val_sampler_desc,
            )

    def on_batch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        batch_idx: int,
        batch: Dict[str, torch.Tensor],
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not trainer.is_rank_zero() or not trainer.model.training:
            return
        if batch_idx % self.log_every_n_batches != 0:
            return

        grad_norms = self._grad_norms(trainer._unwrap_model())
        metric_parts = [f"loss={loss:.4f}"]
        if metrics:
            metric_parts.extend([f"{k}={v:.4f}" for k, v in metrics.items() if v is not None])
        if grad_norms is not None:
            metric_parts.append(
                "grad_norm(total={total:.4f}, max={max:.4f})".format(
                    total=grad_norms["total"], max=grad_norms["max"]
                )
            )

        logging.info(
            "[Debug] Epoch %d | Batch %d -> %s", epoch + 1, batch_idx + 1, ", ".join(metric_parts)
        )

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.is_rank_zero():
            return
        if not metrics:
            logging.info("[Debug] Epoch %d end | no metrics available", epoch + 1)
            return
        formatted = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logging.info("[Debug] Epoch %d complete | %s", epoch + 1, formatted)
