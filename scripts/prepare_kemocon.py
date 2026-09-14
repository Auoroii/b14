"""Build the portable K-EmoCon manifest configured for this project."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SOURCE = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_LOCAL_SOURCE))

from emotion_model.data import build_kemocon_manifest, write_manifest_json
from emotion_model.experiments import (
    load_kemocon_experiment_config,
    resolve_project_relative,
)

def _relative_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "paths must be project-relative and cannot contain '..'."
        )
    return path


def main() -> None:
    """Create a validated manifest without copying or modifying raw data."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=_relative_cli_path,
        default=Path("configs/kemocon_v4_2_full_window_relation_differential.yaml"),
    )
    arguments = parser.parse_args()
    config_path = resolve_project_relative(_PROJECT_ROOT, arguments.config)
    config = load_kemocon_experiment_config(config_path)
    data_root = resolve_project_relative(_PROJECT_ROOT, config.paths.data_root)
    destination = resolve_project_relative(_PROJECT_ROOT, config.paths.manifest)
    report = build_kemocon_manifest(
        data_root,
        annotation_perspective=config.dataset.annotation_perspective,
        channel_names=config.dataset.channel_names,
        window_seconds=config.dataset.window_seconds,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_json(report.manifest, destination)
    print(
        json.dumps(
            {
                "manifest": config.paths.manifest.as_posix(),
                "participants": report.participant_count,
                "sessions": report.session_count,
                "records": report.record_count,
                "skipped_annotation_rows": report.skipped_annotation_rows,
                "skipped_tail_windows": report.skipped_tail_windows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
