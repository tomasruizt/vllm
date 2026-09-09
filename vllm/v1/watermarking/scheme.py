# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.config.watermarking import (
    SpeculativeWatermarkPolicy,
    WatermarkConfig,
    WatermarkRole,
    derive_watermark_key,
)
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.watermarker import Watermarker


class WatermarkKeySchedule(ABC):
    @abstractmethod
    def key_for(self, role: WatermarkRole) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class FixedWatermarkKeySchedule(WatermarkKeySchedule):
    key: int

    def key_for(self, role: WatermarkRole) -> int:
        return self.key


@dataclass(frozen=True)
class DomainSeparatedWatermarkKeySchedule(WatermarkKeySchedule):
    key: int
    generation_role: WatermarkRole = WatermarkRole.TARGET

    def key_for(self, role: WatermarkRole) -> int:
        if role is WatermarkRole.GENERATION:
            role = self.generation_role
        return derive_watermark_key(self.key, role.value.encode())


class WatermarkScheme(ABC):
    @property
    @abstractmethod
    def speculative_policy(self) -> SpeculativeWatermarkPolicy | None:
        raise NotImplementedError

    @abstractmethod
    def watermarker_for(
        self, role: WatermarkRole = WatermarkRole.GENERATION
    ) -> Watermarker:
        raise NotImplementedError


class GumbelWatermarkScheme(WatermarkScheme):
    def __init__(
        self,
        config: WatermarkConfig,
        key_schedule: WatermarkKeySchedule,
    ) -> None:
        self.config = config
        self.key_schedule = key_schedule

    @property
    def speculative_policy(self) -> SpeculativeWatermarkPolicy | None:
        return self.config.speculative_decoding_policy

    def watermarker_for(
        self, role: WatermarkRole = WatermarkRole.GENERATION
    ) -> Watermarker:
        return GumbelWatermarker(
            self.key_schedule.key_for(role),
            self.config.context_width,
            self.config.prf,
        )
