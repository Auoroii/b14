"""Package import smoke tests."""

import importlib
import re

import emotion_model


def test_top_level_package_import_and_version() -> None:
    """Verify the package imports and exposes a semantic version string."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", emotion_model.__version__)


def test_subpackages_import() -> None:
    """Verify all stage-zero subpackages can be imported."""
    subpackages = (
        "common",
        "speech",
        "physiology",
        "multimodal",
        "data",
        "training",
        "evaluation",
    )

    for subpackage in subpackages:
        imported = importlib.import_module(f"emotion_model.{subpackage}")
        assert imported.__name__ == f"emotion_model.{subpackage}"
