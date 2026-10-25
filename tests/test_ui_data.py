"""UI bundle helpers: Sun direction alignment for the Tauri viewer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_ui_data():
    spec = importlib.util.spec_from_file_location(
        "make_ui_data", ROOT / "scripts" / "make_ui_data.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sun_track_is_aligned_and_unit_length(tmp_path):
    path = tmp_path / "sun.csv"
    path.write_text("t,x,y,z\n0,3,4,0\n10,0,0,2\n")
    directions, ranges = _make_ui_data()._load_sun_track([0.0, 10.0], path=path)
    assert directions == [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]
    # Ranges are reported in kilometres even though the CSV is in metres.
    assert ranges == pytest.approx([5.0e-3, 2.0e-3])


def test_sun_track_rejects_unaligned_arc(tmp_path):
    path = tmp_path / "sun.csv"
    path.write_text("t,x,y,z\n0,1,0,0\n")
    assert _make_ui_data()._load_sun_track([0.0, 10.0, 20.0], path=path) is None


def test_sun_track_fills_small_gaps(tmp_path):
    path = tmp_path / "sun.csv"
    path.write_text("t,x,y,z\n0,1,0,0\n20,0,1,0\n")
    directions, ranges = _make_ui_data()._load_sun_track([0.0, 10.0, 20.0], path=path)
    assert directions[1] == directions[0]
    assert ranges[1] == ranges[0]


def test_sun_track_missing_file_returns_none(tmp_path):
    assert _make_ui_data()._load_sun_track([0.0], path=tmp_path / "missing.csv") is None
