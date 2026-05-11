# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

from loaders.dataloader import URGENT2Dataset
from loaders.dataloader_paired import PairedSpeechDataset


def get_dataset_class(dataset_type):
    dataset_type = (dataset_type or "urgent2").lower()
    if dataset_type == "urgent2":
        return URGENT2Dataset
    if dataset_type == "paired":
        return PairedSpeechDataset
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")
