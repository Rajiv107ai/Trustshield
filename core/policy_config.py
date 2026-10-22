"""Single source of truth for serving policy thresholds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskPolicy:
    review: float = 0.25
    hold: float = 0.55
    block: float = 0.85
    version: str = "policy.v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.review <= self.hold <= self.block <= 1.0:
            raise ValueError("risk policy thresholds must be ordered in [0, 1]")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "review": self.review,
            "hold": self.hold,
            "block": self.block,
            "version": self.version,
        }


RISK_POLICY = RiskPolicy()


__all__ = ["RISK_POLICY", "RiskPolicy"]
