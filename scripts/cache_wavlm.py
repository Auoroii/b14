"""Cache the official WavLM base checkpoint at the configured relative path."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import WavLMModel

from emotion_model.experiments import (
    load_kemocon_experiment_config,
    resolve_project_relative,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _relative_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "paths must be project-relative and cannot contain '..'."
        )
    return path


def main() -> None:
    """Download WavLM only when explicitly invoked and save it locally."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=_relative_cli_path,
        default=Path("configs/kemocon_v4_2_full_window_relation_differential.yaml"),
    )
    parser.add_argument("--model-id", default="microsoft/wavlm-base")
    arguments = parser.parse_args()
    config = load_kemocon_experiment_config(
        resolve_project_relative(_PROJECT_ROOT, arguments.config)
    )
    destination = resolve_project_relative(
        _PROJECT_ROOT,
        config.paths.wavlm_model,
    )
    destination.mkdir(parents=True, exist_ok=True)
    model = WavLMModel.from_pretrained(arguments.model_id)
    model.save_pretrained(destination)
    print(f"saved {arguments.model_id} to {config.paths.wavlm_model.as_posix()}")


if __name__ == "__main__":
    main()
