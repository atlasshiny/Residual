"""Pydantic-verified configuration loaded from YAML for the PINN pipeline.

Usage:
    from ml.pinn.config import PipelineConfig
    cfg = PipelineConfig.from_yaml("configs/pinn.yaml")
    # or with overrides:
    cfg = PipelineConfig.from_yaml("configs/pinn.yaml", overrides={"epochs": 200})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError as exc:  # pragma: no cover
    raise ImportError("pydantic is required: pip install pydantic") from exc


class PipelineConfig(BaseModel):
    data: str | None = None
    features: str | None = None
    targets: str | None = None
    mission: str | None = None
    time_col: str | None = None
    split: Literal["random", "time", "group"] = "random"
    group_col: str = "arc_id"
    seed: int = 0
    test_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    epochs: int = Field(default=150, ge=1)
    lr: float = Field(default=5e-2, gt=0.0)
    engineer: bool = False
    out: str = "ml/pinn/pinn_result.json"

    @field_validator("test_fraction")
    @classmethod
    def _tf(cls, v):
        if v < 0 or v >= 1:
            raise ValueError("test_fraction must be in [0, 1)")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None):
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if overrides:
            raw = {**raw, **overrides}
        return cls.model_validate(raw)

    def to_argv(self) -> list[str]:
        argv = []
        if self.data:
            argv += ["--data", self.data]
        if self.features:
            argv += ["--features", self.features]
        if self.targets:
            argv += ["--targets", self.targets]
        if self.mission:
            argv += ["--mission", self.mission]
        if self.time_col:
            argv += ["--time-col", self.time_col]
        argv += ["--split", self.split]
        argv += ["--group-col", self.group_col]
        argv += ["--seed", str(self.seed)]
        argv += ["--test-fraction", str(self.test_fraction)]
        argv += ["--epochs", str(self.epochs)]
        argv += ["--lr", str(self.lr)]
        if self.engineer:
            argv += ["--engineer"]
        argv += ["--out", self.out]
        return argv
