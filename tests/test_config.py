"""Tests for ml.kan.config.PipelineConfig."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pydantic = pytest.importorskip("pydantic")

from ml.kan.config import PipelineConfig


def test_default_valid():
    c = PipelineConfig()
    assert c.test_fraction == 0.2
    assert c.epochs == 150


def test_from_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("data: x.csv\ntest_fraction: 0.3\nepochs: 10\nengineer: true\n")
    c = PipelineConfig.from_yaml(p)
    assert c.data == "x.csv"
    assert c.test_fraction == 0.3
    assert c.engineer is True
    assert "--engineer" in c.to_argv()


def test_validation_rejects_bad():
    with pytest.raises(Exception):
        PipelineConfig(test_fraction=1.5)
    with pytest.raises(Exception):
        PipelineConfig(epochs=0)
    with pytest.raises(Exception):
        PipelineConfig(lr=0.0)


def test_overrides():
    p = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    p.write("epochs: 10\n")
    p.close()
    c = PipelineConfig.from_yaml(p.name, overrides={"epochs": 99})
    assert c.epochs == 99
    Path(p.name).unlink()
