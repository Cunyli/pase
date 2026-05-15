# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

from loaders.dataloader_use_simulation import (
    UseSimulationCleanSpeechDataset,
    UseSimulationFixedPairSpeechDataset,
    UseSimulationOnlineSpeechDataset,
)


def get_dataset_class(dataset_type):
    dataset_type = (dataset_type or "urgent2").lower()
    if dataset_type == "use_simulation_clean":
        return UseSimulationCleanSpeechDataset
    if dataset_type in ("use_simulation", "use_simulation_onthefly"):
        return UseSimulationOnlineSpeechDataset
    if dataset_type == "use_simulation_fixed":
        return UseSimulationFixedPairSpeechDataset
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")
