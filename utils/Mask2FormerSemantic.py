"""Mask2Former wrapper for native mask-query training and dense inference."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


DEFAULT_ID2LABEL = {
    0: "wing",
    1: "envelope",
    2: "seed",
}
DEFAULT_LABEL2ID = {label: idx for idx, label in DEFAULT_ID2LABEL.items()}


class Mask2FormerSemantic(torch.nn.Module):
    """Use Mask2Former's native loss, with dense logits for tiled inference."""

    def __init__(
        self,
        pretrained_model_name: str = "facebook/mask2former-swin-large-ade-semantic",
        num_labels: int = 3,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        pretrained: bool = True,
        ignore_mismatched_sizes: bool = True,
        config_dict: dict | None = None,
        local_files_only: bool = False,
        semantic_score_eps: float = 1e-6,
    ):
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

        self.num_labels = num_labels
        self.semantic_score_eps = semantic_score_eps

        id2label = id2label or DEFAULT_ID2LABEL
        label2id = label2id or DEFAULT_LABEL2ID

        if pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                local_files_only=local_files_only,
            )
        else:
            if config_dict is None:
                config = Mask2FormerConfig.from_pretrained(
                    pretrained_model_name,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    local_files_only=local_files_only,
                )
            else:
                config = Mask2FormerConfig.from_dict(copy.deepcopy(config_dict))
                config.num_labels = num_labels
                config.id2label = id2label
                config.label2id = label2id
            self.model = Mask2FormerForUniversalSegmentation(config)

    def dense_semantic_logits(self, outputs, output_size: tuple[int, int]) -> torch.Tensor:
        """Convert query class/mask outputs to [background, wing, envelope, seed]."""
        class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., : self.num_labels]
        mask_probs = outputs.masks_queries_logits.sigmoid()
        foreground_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        background_scores = 1.0 - foreground_scores.amax(dim=1, keepdim=True)
        semantic_scores = torch.cat(
            [background_scores.clamp_min(self.semantic_score_eps), foreground_scores],
            dim=1,
        )
        semantic_logits = torch.log(semantic_scores.clamp_min(self.semantic_score_eps))

        if semantic_logits.shape[-2:] != output_size:
            semantic_logits = F.interpolate(
                semantic_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return semantic_logits

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
        return_outputs: bool = False,
    ) -> torch.Tensor:
        if (mask_labels is None) != (class_labels is None):
            raise ValueError("mask_labels and class_labels must be provided together.")

        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        if mask_labels is not None:
            return outputs if return_outputs else outputs.loss

        return self.dense_semantic_logits(outputs, pixel_values.shape[-2:])
