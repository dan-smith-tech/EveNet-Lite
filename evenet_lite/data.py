from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist

from .callbacks import EvenetLiteNormalizer


class EvenetTensorDataset(Dataset):
    """Dataset wrapper around in-memory tensors.

    Args:
        features: Mapping of feature group name to tensor with leading batch dimension.
        labels: Target labels tensor.
        sample_weights: Optional per-sample weights tensor.
        normalizer: Normalizer applied on-the-fly when retrieving items.
    """

    def __init__(
        self,
        features: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        normalizer: Optional[EvenetLiteNormalizer] = None,
    ) -> None:
        self.raw_features = {k: torch.as_tensor(v) for k, v in features.items()}
        self.labels = torch.as_tensor(labels)
        self.sample_weights = torch.as_tensor(sample_weights) if sample_weights is not None else None
        self.normalizer = normalizer

    def set_normalizer(self, normalizer: EvenetLiteNormalizer) -> None:
        self.normalizer = normalizer

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        features = {k: v[idx] for k, v in self.raw_features.items()}
        if self.normalizer is not None:
            features = self.normalizer.transform({k: v.unsqueeze(0) for k, v in features.items()})
            features = {k: v.squeeze(0) for k, v in features.items()}
        label = self.labels[idx]
        weight = self.sample_weights[idx] if self.sample_weights is not None else torch.tensor(torch.inf)
        return features, label, weight


class DistributedWeightedSampler(Sampler[int]):
    """Distributed-aware weighted sampler.

    Generates a global weighted sample list and shards it across ranks so that
    each replica processes a distinct subset. Sampling is with replacement to
    mirror ``WeightedRandomSampler`` semantics.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        epoch_size: Optional[int] = None,
    ) -> None:
        if weights.dim() != 1:
            raise ValueError("weights should be a 1D tensor")
        self.weights = weights.float()
        self.replacement = replacement
        self.epoch_size = int(epoch_size) if epoch_size is not None else len(weights)
        self.num_samples = num_samples if num_samples is not None else self.epoch_size

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        self.num_samples_per_replica = (self.num_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples_per_replica * self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=generator).tolist()
        # Subsample for this replica
        offset_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(offset_indices)

    def __len__(self) -> int:
        return self.num_samples_per_replica


def build_sampler(
    sampler: Optional[str],
    dataset: EvenetTensorDataset,
    weights: Optional[torch.Tensor],
    epoch_size: Optional[int] = None,
) -> Optional[Sampler[int]]:
    if sampler == "weighted":
        if weights is None:
            # derive weights from labels
            labels = dataset.labels
            class_counts = torch.bincount(labels.long())
            class_weights = class_counts.float().reciprocal().clamp_max(class_counts.numel())
            sample_weights = class_weights[labels.long()]
        else:
            sample_weights = torch.as_tensor(weights).float()
        return DistributedWeightedSampler(sample_weights, epoch_size=epoch_size)
    return None
