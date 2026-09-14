"""Offline contracts for optional K-EmoCon checkpoint ablation evaluation."""

from __future__ import annotations

import pytest

from scripts.evaluate_kemocon import _resolve_ablation_enabled


@pytest.mark.parametrize(
    ("configured", "forced_by_cli", "expected"),
    (
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ),
)
def test_resolve_ablation_combines_config_and_cli(
    configured: bool,
    forced_by_cli: bool,
    expected: bool,
) -> None:
    """Allow the CLI to enable an otherwise disabled multimodal ablation."""

    assert (
        _resolve_ablation_enabled(
            configured=configured,
            forced_by_cli=forced_by_cli,
        )
        is expected
    )
