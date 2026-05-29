"""Dense pennycress segmentation head on top of a LoRA-tuned SAM3 encoder."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_ID2LABEL = {
    0: "wing",
    1: "envelope",
    2: "seed",
}
DEFAULT_LABEL2ID = {label: idx for idx, label in DEFAULT_ID2LABEL.items()}
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_LORA_TARGET_KEYWORDS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "query",
    "key",
    "value",
    "dense",
    "fc1",
    "fc2",
)


class ConvNormAct(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.GELU(),
        )


class _Sam3VisionFeatureWrapper(torch.nn.Module):
    """Fallback wrapper when Transformers exposes SAM3 features only on Sam3Model."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)

    def forward(self, pixel_values: torch.Tensor) -> Any:
        return self.model.get_vision_features(pixel_values=pixel_values)


def _module_name_suffix(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _as_config_dict(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, dict):
        return copy.deepcopy(config)
    return None


def _set_image_size(config: Any, image_size: int | None) -> None:
    if image_size is None:
        return

    seen: set[int] = set()

    def visit(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))

        if isinstance(obj, dict):
            if "image_size" in obj:
                obj["image_size"] = image_size
            for key in ("vision_config", "backbone_config"):
                visit(obj.get(key))
            return

        if hasattr(obj, "image_size"):
            setattr(obj, "image_size", image_size)
        for key in ("vision_config", "backbone_config"):
            visit(getattr(obj, key, None))

    visit(config)


def _iter_nested_configs(config: Any) -> Iterable[Any]:
    if config is None:
        return

    yield config
    if isinstance(config, dict):
        for key in ("vision_config", "backbone_config", "config"):
            yield from _iter_nested_configs(config.get(key))
    else:
        for key in ("vision_config", "backbone_config", "config"):
            yield from _iter_nested_configs(getattr(config, key, None))


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


class Sam3Custom(torch.nn.Module):
    """SAM3 vision features plus a dense decoder for pennycress masks.

    The wrapper returns logits with four channels:
    background, wing, envelope, and seed.  Pretrained SAM3 parameters are
    frozen by PEFT and adapted through LoRA; the dense decoder is trained
    normally by the repo's existing cross-entropy losses.
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/sam3",
        num_labels: int = 3,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        pretrained: bool = True,
        ignore_mismatched_sizes: bool = True,
        config_dict: dict[str, Any] | None = None,
        local_files_only: bool = False,
        use_lora: bool = True,
        peft: bool | None = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        lora_target_keywords: tuple[str, ...] = DEFAULT_LORA_TARGET_KEYWORDS,
        encoder_image_size: int | None = 560,
        decoder_channels: int = 256,
        image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
        image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
    ):
        super().__init__()

        if peft is not None:
            use_lora = peft

        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.num_classes = num_labels + 1
        self.id2label = id2label or DEFAULT_ID2LABEL
        self.label2id = label2id or DEFAULT_LABEL2ID
        self.encoder_image_size = encoder_image_size
        self.decoder_channels = decoder_channels
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.ignore_mismatched_sizes = ignore_mismatched_sizes

        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.encoder, self.config_dict = self._build_encoder(
            pretrained=pretrained,
            config_dict=config_dict,
            local_files_only=local_files_only,
        )
        if use_lora:
            self.encoder = self._apply_lora(
                self.encoder,
                target_modules=lora_target_modules,
                target_keywords=lora_target_keywords,
            )

        hidden_size = self._infer_feature_channels(self.encoder, self.config_dict)
        self.decoder = torch.nn.Sequential(
            ConvNormAct(hidden_size, decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels),
            torch.nn.Conv2d(decoder_channels, self.num_classes, kernel_size=1),
        )

    def _build_encoder(
        self,
        pretrained: bool,
        config_dict: dict[str, Any] | None,
        local_files_only: bool,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        try:
            from transformers import Sam3Config, Sam3Model
        except ImportError as exc:
            raise ImportError(
                "SAM3 requires a Transformers build that exposes Sam3Model "
                "(Transformers 5.x or newer in current Hugging Face docs)."
            ) from exc

        if pretrained:
            config = None
            if self.encoder_image_size is not None:
                config = Sam3Config.from_pretrained(
                    self.pretrained_model_name,
                    local_files_only=local_files_only,
                )
                _set_image_size(config, self.encoder_image_size)
            sam3 = Sam3Model.from_pretrained(
                self.pretrained_model_name,
                config=config,
                ignore_mismatched_sizes=self.ignore_mismatched_sizes,
                local_files_only=local_files_only,
            )
        else:
            if config_dict is None:
                config = Sam3Config.from_pretrained(
                    self.pretrained_model_name,
                    local_files_only=local_files_only,
                )
            else:
                config = Sam3Config.from_dict(copy.deepcopy(config_dict))
            _set_image_size(config, self.encoder_image_size)
            sam3 = Sam3Model(config)

        encoder = self._extract_vision_encoder(sam3)
        full_config_dict = _as_config_dict(getattr(sam3, "config", None)) or {}
        return encoder, full_config_dict

    def _extract_vision_encoder(self, sam3: torch.nn.Module) -> torch.nn.Module:
        for attr in ("vision_model", "vision_encoder", "image_encoder", "vision_tower"):
            encoder = getattr(sam3, attr, None)
            if encoder is not None:
                return encoder

        if hasattr(sam3, "get_vision_features"):
            return _Sam3VisionFeatureWrapper(sam3)

        raise AttributeError(
            "Could not find a SAM3 vision encoder on the loaded model. "
            "Inspect the installed Transformers SAM3 attribute names and update Sam3Custom."
        )

    def _apply_lora(
        self,
        encoder: torch.nn.Module,
        target_modules: list[str] | None,
        target_keywords: tuple[str, ...],
    ) -> torch.nn.Module:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "SAM3 LoRA training requires the `peft` package in the active environment."
            ) from exc

        if target_modules is None:
            target_modules = self._discover_lora_targets(encoder, target_keywords)

        if not target_modules:
            raise ValueError("No Linear modules were found for SAM3 LoRA injection.")

        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        return get_peft_model(encoder, config)

    def _discover_lora_targets(
        self,
        module: torch.nn.Module,
        keywords: tuple[str, ...],
    ) -> list[str]:
        linear_suffixes = {
            _module_name_suffix(name)
            for name, child in module.named_modules()
            if isinstance(child, torch.nn.Linear)
        }
        preferred = sorted(
            suffix
            for suffix in linear_suffixes
            if any(keyword in suffix for keyword in keywords)
        )
        return preferred or sorted(linear_suffixes)

    def _infer_feature_channels(
        self,
        encoder: torch.nn.Module,
        config_dict: dict[str, Any],
    ) -> int:
        for config in _iter_nested_configs(getattr(encoder, "config", None)):
            for key in ("fpn_hidden_size", "hidden_size", "embed_dim", "num_features"):
                value = _config_value(config, key)
                if isinstance(value, int):
                    return value

        for config in _iter_nested_configs(config_dict):
            for key in ("fpn_hidden_size", "hidden_size", "embed_dim", "num_features"):
                value = _config_value(config, key)
                if isinstance(value, int):
                    return value

        raise ValueError(
            "Could not infer SAM3 feature channels from config. "
            "Pass a SAM3 config with fpn_hidden_size/hidden_size or update Sam3Custom."
        )

    def _normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return (pixel_values - self.image_mean.to(pixel_values)) / self.image_std.to(pixel_values)

    def _encoder_forward(self, pixel_values: torch.Tensor) -> Any:
        try:
            return self.encoder(pixel_values=pixel_values)
        except TypeError:
            return self.encoder(pixel_values)

    def _flatten_output_tensors(self, value: Any) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            tensors = []
            for item in value.values():
                tensors.extend(self._flatten_output_tensors(item))
            return tensors
        if isinstance(value, (list, tuple)):
            tensors = []
            for item in value:
                tensors.extend(self._flatten_output_tensors(item))
            return tensors
        return []

    def _sequence_to_map(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = features.shape
        side = int(math.sqrt(sequence_length))
        if side * side != sequence_length:
            side = int(math.sqrt(sequence_length - 1))
            if side * side == sequence_length - 1:
                features = features[:, 1:, :]
            else:
                raise ValueError(
                    f"Cannot reshape SAM3 sequence length {sequence_length} into a square feature map."
                )
        return features.transpose(1, 2).reshape(batch_size, channels, side, side)

    def _feature_map_from_outputs(self, outputs: Any) -> torch.Tensor:
        for attr in (
            "feature_maps",
            "features",
            "image_embeddings",
            "image_embed",
            "embeddings",
            "last_hidden_state",
        ):
            if hasattr(outputs, attr):
                tensors = self._flatten_output_tensors(getattr(outputs, attr))
                selected = self._select_feature_tensor(tensors)
                if selected is not None:
                    return selected

        selected = self._select_feature_tensor(self._flatten_output_tensors(outputs))
        if selected is not None:
            return selected

        raise ValueError("SAM3 encoder output did not contain a usable tensor feature map.")

    def _select_feature_tensor(self, tensors: list[torch.Tensor]) -> torch.Tensor | None:
        tensors = [tensor for tensor in tensors if tensor.ndim in (3, 4)]
        if not tensors:
            return None

        four_dimensional = [tensor for tensor in tensors if tensor.ndim == 4]
        if four_dimensional:
            return max(four_dimensional, key=lambda tensor: tensor.shape[-2] * tensor.shape[-1])

        return self._sequence_to_map(tensors[-1])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output_size = pixel_values.shape[-2:]
        encoder_inputs = self._normalize(pixel_values)
        if self.encoder_image_size is not None and output_size != (
            self.encoder_image_size,
            self.encoder_image_size,
        ):
            encoder_inputs = F.interpolate(
                encoder_inputs,
                size=(self.encoder_image_size, self.encoder_image_size),
                mode="bilinear",
                align_corners=False,
            )

        outputs = self._encoder_forward(encoder_inputs)
        features = self._feature_map_from_outputs(outputs)
        logits = self.decoder(features)

        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

    def trainable_parameter_counts(self) -> tuple[int, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return trainable, total


sam3Custom = Sam3Custom
