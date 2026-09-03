"""Versioned task-independent verification thresholds."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math


VERIFICATION_POLICY_VERSION = 1


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Central measurement thresholds that planning attempts cannot change."""

    version: int = VERIFICATION_POLICY_VERSION
    position_tolerance_m: float = 0.02
    orientation_tolerance_rad: float = 0.10
    region_boundary_tolerance_m: float = 0.005
    support_height_tolerance_m: float = 0.015
    contact_force_min_n: float = 0.2
    contact_duration_s: float = 0.05
    attachment_duration_s: float = 0.10
    attachment_position_tolerance_m: float = 0.02
    attachment_velocity_tolerance_mps: float = 0.05
    lift_displacement_m: float = 0.04
    release_separation_m: float = 0.04
    release_duration_s: float = 0.10
    settled_linear_velocity_mps: float = 0.03
    settled_angular_velocity_radps: float = 0.10
    settled_duration_s: float = 0.20
    base_position_tolerance_m: float = 0.05
    base_yaw_tolerance_rad: float = 0.10
    max_explicit_tolerance_m: float = 0.10

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or self.version != VERIFICATION_POLICY_VERSION:
            raise ValueError(
                f"verification policy version must be {VERIFICATION_POLICY_VERSION}"
            )
        for item in fields(self):
            if item.name == "version":
                continue
            value = getattr(self, item.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"verification policy {item.name} must be finite and positive")


__all__ = ["VERIFICATION_POLICY_VERSION", "VerificationPolicy"]
