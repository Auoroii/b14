"""Traverse prepared K-EmoCon windows without loading WavLM."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SOURCE = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_LOCAL_SOURCE))

from torch.utils.data import DataLoader

from emotion_model.data import (
    KEmoConPhysioAdapter,
    collate_aligned_multimodal_samples,
    partition_manifest_by_participant,
    read_manifest_json,
    summarize_speech_activity_ratios,
)
from emotion_model.experiments import (
    build_configured_kemocon_split,
    build_kemocon_dataset,
    fit_kemocon_train_normalizer,
    load_kemocon_experiment_config,
    resolve_project_relative,
)
from emotion_model.physiology import ChannelwiseZScoreNormalizer

def _relative_cli_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "paths must be project-relative and cannot contain '..'."
        )
    return path


def main() -> None:
    """Validate every Dataset/Collate output and report availability counts."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=_relative_cli_path,
        default=Path("configs/kemocon_v4_2_full_window_relation_differential.yaml"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.workers < 0:
        raise ValueError("workers must be non-negative.")
    config = load_kemocon_experiment_config(
        resolve_project_relative(_PROJECT_ROOT, arguments.config)
    )
    manifest = read_manifest_json(
        resolve_project_relative(_PROJECT_ROOT, config.paths.manifest)
    )
    if not 0 <= arguments.fold < config.split.run_count:
        raise ValueError("fold index lies outside configured runs.")
    split = build_configured_kemocon_split(
        manifest,
        config.split,
        fold_index=arguments.fold,
        label_protocol=config.dataset.label_protocol,
    )
    partitioned = partition_manifest_by_participant(
        manifest,
        split,
    )
    data_root = resolve_project_relative(_PROJECT_ROOT, config.paths.data_root)
    normalizer = fit_kemocon_train_normalizer(
        partitioned.train_records,
        manifest.channel_specs,
        KEmoConPhysioAdapter(data_root),
    )
    partitions = (
        ("train", partitioned.train_records),
        ("validation", partitioned.validation_records),
        ("test", partitioned.test_records),
    )
    all_counts = {
        "records": 0,
        "speech": 0,
        "physiology": 0,
        "ignored_arousal": 0,
        "ignored_valence": 0,
        "speech_source_unavailable": 0,
    }
    all_activity_ratios: list[float] = []
    all_participant_ids: set[str] = set()
    all_session_ids: set[str] = set()
    for name, records in partitions:
        dataset = build_kemocon_dataset(
            records,
            manifest.channel_specs,
            data_root=data_root,
            config=config.dataset,
            normalizer=normalizer,
            modality_mode=config.training.modality_mode,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=arguments.workers,
            collate_fn=collate_aligned_multimodal_samples,
            persistent_workers=arguments.workers > 0,
        )
        counts = {
            "records": 0,
            "speech": 0,
            "physiology": 0,
            "ignored_arousal": 0,
            "ignored_valence": 0,
            "speech_source_unavailable": 0,
        }
        activity_ratios: list[float] = []
        for batch in loader:
            counts["records"] += len(batch.records)
            counts["speech"] += int(batch.speech_available.sum().item())
            counts["physiology"] += int(
                batch.physiology_available.sum().item()
            )
            counts["ignored_arousal"] += int(
                (batch.arousal_labels == -100).sum().item()
            )
            counts["ignored_valence"] += int(
                (batch.valence_labels == -100).sum().item()
            )
            source_present = torch.tensor(
                [record.speech_source is not None for record in batch.records],
                dtype=torch.bool,
            )
            counts["speech_source_unavailable"] += int(
                (~source_present).sum().item()
            )
            if config.dataset.speech_activity.enabled:
                if batch.speech_activity_ratios is None:
                    raise RuntimeError("speech activity diagnostics are missing.")
                activity_ratios.extend(
                    float(value)
                    for value in batch.speech_activity_ratios[source_present].tolist()
                )
        activity_summary = (
            summarize_speech_activity_ratios(activity_ratios)
            if activity_ratios
            else None
        )
        for count_name, value in counts.items():
            all_counts[count_name] += value
        all_activity_ratios.extend(activity_ratios)
        all_participant_ids.update(record.participant_id for record in records)
        all_session_ids.update(record.session_id for record in records)
        print(
            json.dumps(
                {
                    "partition": name,
                    "participant_ids": sorted(
                        {record.participant_id for record in records},
                        key=lambda value: int(value[1:]),
                    ),
                    "session_ids": sorted(
                        {record.session_id for record in records}
                    ),
                    **counts,
                    "speech_activity": activity_summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "partition": "all",
                "participant_ids": sorted(
                    all_participant_ids,
                    key=lambda value: int(value[1:]),
                ),
                "session_ids": sorted(all_session_ids),
                **all_counts,
                "speech_activity": (
                    summarize_speech_activity_ratios(all_activity_ratios)
                    if all_activity_ratios
                    else None
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
