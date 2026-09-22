"""Registry for local Jev-style model families and adapters."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
from collections.abc import Mapping, Sequence
from .schema import Question
from .encoder_model import EncoderDecisionModel
from .local_model import LocalDecisionModel


class DecisionBackend(Protocol):
    def decide(
        self, state: str | Mapping[str, Any] | Sequence[Any], question: Question
    ) -> dict[str, Any]: ...


class BackendKind(str, Enum):
    CAUSAL = "causal"
    ENCODER = "encoder"
    SGLANG = "sglang"


@dataclass(frozen=True)
class BackendSpec:
    model_id: str
    kind: BackendKind
    source_repo: str = "user"
    notes: str = ""


BACKENDS: dict[str, BackendSpec] = {
    "Qwen/Qwen2.5-0.5B-Instruct": BackendSpec(
        "Qwen/Qwen2.5-0.5B-Instruct",
        BackendKind.CAUSAL,
        "zhihz/openjev",
        "local next-token readout",
    ),
    "Qwen/Qwen3-0.6B": BackendSpec(
        "Qwen/Qwen3-0.6B", BackendKind.CAUSAL, "zhihz/openjev", "small local baseline"
    ),
    "Qwen/Qwen3-4B-Instruct-2507": BackendSpec(
        "Qwen/Qwen3-4B-Instruct-2507",
        BackendKind.CAUSAL,
        "zhihz/openjev",
        "local bilingual readout",
    ),
    "Qwen/Qwen3.6-35B-A3B": BackendSpec(
        "Qwen/Qwen3.6-35B-A3B",
        BackendKind.SGLANG,
        "ekzhang/openjev-sglang",
        "local SGLang open model",
    ),
    "heman10x/rlcd-modernbert-151m": BackendSpec(
        "heman10x/rlcd-modernbert-151m",
        BackendKind.ENCODER,
        "Heman10x-NGU/openJev-verdict-2.0",
        "ModernBERT/GLiClass decision checkpoint",
    ),
    "knowledgator/gliclass-modern-base-v2.0": BackendSpec(
        "knowledgator/gliclass-modern-base-v2.0",
        BackendKind.ENCODER,
        "Heman10x-NGU/openJev-verdict-2.0",
        "GLiClass encoder route",
    ),
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4": BackendSpec(
        "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        BackendKind.SGLANG,
        "razorback16/openjev",
        "local vLLM/SGLang or MLX service",
    ),
}


def register_backend(spec: BackendSpec) -> None:
    if not spec.model_id.strip():
        raise ValueError("backend model_id cannot be empty")
    BACKENDS[spec.model_id] = spec


def list_backends() -> tuple[BackendSpec, ...]:
    return tuple(BACKENDS.values())


def load_backend(
    model_id: str,
    *,
    kind: BackendKind | str | None = None,
    endpoint: str | None = None,
    device: str = "auto",
    **kwargs: Any,
) -> DecisionBackend:
    spec = BACKENDS.get(model_id)
    selected_kind = BackendKind(kind) if kind is not None else (spec.kind if spec else None)
    if selected_kind is None:
        raise ValueError(
            f"unknown model id: {model_id}; pass kind='causal', 'encoder', or 'sglang'"
        )
    if selected_kind is BackendKind.CAUSAL:
        return LocalDecisionModel.from_pretrained(model_id, device=device, **kwargs)
    if selected_kind is BackendKind.ENCODER:
        return EncoderDecisionModel.from_pretrained(model_id, device=device, **kwargs)
    if endpoint is None:
        raise ValueError("SGLang/DiffusionGemma backends require a local endpoint")
    from .sglang_backend import SGLangDecisionModel

    return SGLangDecisionModel(endpoint, model_id=model_id)
