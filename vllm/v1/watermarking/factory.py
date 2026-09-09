# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.watermarking import WatermarkConfig, WatermarkRole
from vllm.v1.watermarking.scheme import (
    DomainSeparatedWatermarkKeySchedule,
    FixedWatermarkKeySchedule,
    GumbelWatermarkScheme,
    WatermarkKeySchedule,
    WatermarkScheme,
)
from vllm.v1.watermarking.watermarker import Watermarker


def create_watermark_scheme(config: WatermarkConfig) -> WatermarkScheme:
    key_schedule: WatermarkKeySchedule
    if config.algorithm == "gumbel":
        key_schedule = FixedWatermarkKeySchedule(config.key)
    elif config.algorithm == "dual_key_gumbel":
        key_schedule = DomainSeparatedWatermarkKeySchedule(config.key)
    else:
        raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
    return GumbelWatermarkScheme(config, key_schedule)


def create_watermarker(
    config: WatermarkConfig,
    *,
    role: WatermarkRole = WatermarkRole.GENERATION,
) -> Watermarker:
    return create_watermark_scheme(config).watermarker_for(role)
