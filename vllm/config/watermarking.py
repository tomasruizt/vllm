# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

WatermarkingAlgorithm = Literal["gumbel", "dual_key_gumbel"]
WatermarkPRFName = Literal["philox"]


class WatermarkRole(str, Enum):
    GENERATION = "generation"
    DRAFT = "draft"
    TARGET = "target"
    ACCEPTANCE = "acceptance"


class SpeculativeVerification(str, Enum):
    ORDINARY = "ordinary"
    WATERMARKED = "watermarked"


class AcceptanceRandomness(str, Enum):
    RANDOM = "random"
    KEYED = "keyed"


@dataclass(frozen=True)
class SpeculativeWatermarkPolicy:
    draft_role: WatermarkRole
    target_role: WatermarkRole
    verification: SpeculativeVerification
    acceptance_randomness: AcceptanceRandomness


FAST_SPECULATIVE_WATERMARK_POLICY = SpeculativeWatermarkPolicy(
    draft_role=WatermarkRole.DRAFT,
    target_role=WatermarkRole.TARGET,
    verification=SpeculativeVerification.ORDINARY,
    acceptance_randomness=AcceptanceRandomness.RANDOM,
)


def derive_watermark_key(key: int, domain: bytes) -> int:
    digest = hashlib.sha256(domain + key.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


@config
class WatermarkConfig:
    """Configuration for text watermark generation."""

    key: int = Field(ge=0, repr=False, exclude=True)
    """Secret key used to watermark generated text."""
    algorithm: WatermarkingAlgorithm = "gumbel"
    """Algorithm used to watermark generated text."""
    context_width: int = Field(default=4, ge=1)
    """Number of prior output tokens used by the watermark PRF."""
    prf: WatermarkPRFName = "philox"
    """Pseudorandom function used by the watermarking algorithm."""

    @model_validator(mode="after")
    def validate_key(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")
        if self.algorithm == "gumbel":
            logger.warning_once(
                "Single-key Gumbel-max watermarking may increase the frequency of "
                "degenerate generations, including repetition loops.",
                scope="global",
            )
        return self

    @property
    def speculative_decoding_policy(self) -> SpeculativeWatermarkPolicy | None:
        if self.algorithm == "dual_key_gumbel":
            return FAST_SPECULATIVE_WATERMARK_POLICY
        return None
