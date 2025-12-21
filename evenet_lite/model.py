"""EveNet backbone wrapper used by Evenet-Lite trainers.

This module re-exports the reference EveNetLite architecture that stitches
together the upstream `evenet` submodule components (embedding, PET body,
object encoder, and classification head). Users are expected to bring a
populated `evenet` submodule (e.g., via git submodule) so these imports
resolve at runtime.
"""



from typing import List

import torch
from torch import nn

from evenet.control.global_config import DotDict
from evenet.network.body.embedding import GlobalVectorEmbedding, PETBody
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.classification.classification_head import ClassificationHead


class EveNetLite(nn.Module):
    """Thin EveNet wrapper exposing a classification head.

    Parameters
    ----------
    config:
        Parsed EveNet configuration (DotDict) providing Body and Classification
        sections.
    global_input_dim:
        Dimensionality of global (per-event) features.
    sequential_input_dim:
        Dimensionality of sequential/object-level features.
    cls_label:
        List of class labels for the classification head.
    """

    def __init__(
        self,
        config: DotDict,
        global_input_dim: int,
        sequential_input_dim: int,
        cls_label: List[str],
    ) -> None:
        super().__init__()

        self.network_cfg = config
        self.local_feature_indices = self.network_cfg.Body.PET.local_point_index
        self.global_input_dim = global_input_dim
        self.sequential_input_dim = sequential_input_dim
        self.class_label = {"EVENT": cls_label}
        self.num_classes = {"EVENT": len(cls_label)}

        # [1] Global Embedding
        global_embedding_cfg = self.network_cfg.Body.GlobalEmbedding
        self.GlobalEmbedding = GlobalVectorEmbedding(
            linear_block_type=global_embedding_cfg.linear_block_type,
            input_dim=self.global_input_dim,
            hidden_dim_scale=global_embedding_cfg.transformer_dim_scale,
            initial_embedding_dim=global_embedding_cfg.initial_embedding_dim,
            final_embedding_dim=global_embedding_cfg.hidden_dim,
            normalization_type=global_embedding_cfg.normalization,
            activation_type=global_embedding_cfg.linear_activation,
            skip_connection=global_embedding_cfg.skip_connection,
            num_embedding_layers=global_embedding_cfg.num_embedding_layers,
            dropout=global_embedding_cfg.dropout,
        )

        # [2] PET Body
        pet_config = self.network_cfg.Body.PET
        self.PET = PETBody(
            num_feat=self.sequential_input_dim,
            num_keep=pet_config.num_feature_keep,
            feature_drop=pet_config.feature_drop,
            projection_dim=pet_config.hidden_dim,
            local=pet_config.enable_local_embedding,
            K=pet_config.local_Krank,
            num_local=pet_config.num_local_layer,
            num_layers=pet_config.num_layers,
            num_heads=pet_config.num_heads,
            drop_probability=pet_config.drop_probability,
            talking_head=pet_config.talking_head,
            layer_scale=pet_config.layer_scale,
            layer_scale_init=pet_config.layer_scale_init,
            dropout=pet_config.dropout,
            mode=pet_config.mode,
        )

        # [3] Classification + Regression + Assignment Body
        obj_encoder_cfg = self.network_cfg.Body.ObjectEncoder
        self.ObjectEncoder = ObjectEncoder(
            input_dim=pet_config.hidden_dim,
            hidden_dim=obj_encoder_cfg.hidden_dim,
            output_dim=obj_encoder_cfg.hidden_dim,
            position_embedding_dim=obj_encoder_cfg.position_embedding_dim,
            num_heads=obj_encoder_cfg.num_attention_heads,
            transformer_dim_scale=obj_encoder_cfg.transformer_dim_scale,
            num_linear_layers=obj_encoder_cfg.num_embedding_layers,
            num_encoder_layers=obj_encoder_cfg.num_encoder_layers,
            dropout=obj_encoder_cfg.dropout,
            conditioned=False,
            skip_connection=obj_encoder_cfg.skip_connection,
            encoder_skip_connection=obj_encoder_cfg.encoder_skip_connection,
        )

        # [4] Classification Head
        cls_cfg = self.network_cfg.Classification
        self.Classification = ClassificationHead(
            input_dim=obj_encoder_cfg.hidden_dim,
            class_label=self.class_label,
            event_num_classes=self.num_classes,
            num_layers=cls_cfg.num_classification_layers,
            hidden_dim=cls_cfg.hidden_dim,
            skip_connection=cls_cfg.skip_connection,
            dropout=cls_cfg.dropout,
            num_attention_heads=cls_cfg.num_attention_heads,
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        """Apply EveNet backbone and classification head."""

        input_point_cloud = x  # (B, N, D)
        B, _N, _D = input_point_cloud.shape
        input_point_cloud_mask = x_mask.unsqueeze(-1)  # (B, N, 1)
        global_conditions = globals.unsqueeze(1)  # (B, 1, Dg)
        global_conditions_mask = torch.ones((B, 1, 1), device=input_point_cloud.device)  # (B, 1, 1)
        time = torch.zeros((B,), device=input_point_cloud.device)
        full_attn_mask = None
        time_masking = torch.zeros_like(input_point_cloud_mask).float()
        global_feature_mask = torch.ones_like(global_conditions).float()

        full_global_conditions = self.GlobalEmbedding(
            x=global_conditions * global_feature_mask,
            mask=global_conditions_mask,
        )

        local_points = input_point_cloud[..., self.local_feature_indices]
        full_input_point_cloud = self.PET(
            input_features=input_point_cloud,
            input_points=local_points,
            mask=input_point_cloud_mask,
            attn_mask=full_attn_mask,
            time=time,
            time_masking=time_masking,
        )
        embeddings, embedded_global_conditions, event_token = self.ObjectEncoder(
            encoded_vectors=full_input_point_cloud,
            mask=input_point_cloud_mask,
            condition_vectors=full_global_conditions,
            condition_mask=global_conditions_mask,
        )
        classifications = self.Classification(
            x=embeddings,
            x_mask=input_point_cloud_mask,
            event_token=event_token,
        )
        return classifications["classification/EVENT"]


__all__ = ["EveNetLite"]
