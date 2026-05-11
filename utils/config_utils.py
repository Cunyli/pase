# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os

from omegaconf import OmegaConf


def _expand_paths(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    return value


def load_config(path):
    config = OmegaConf.load(os.path.expanduser(os.path.expandvars(path)))
    resolved = OmegaConf.to_container(config, resolve=True)
    return OmegaConf.create(_expand_paths(resolved))
