# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import argparse
from pathlib import Path

import torch

from models.wavlm.feature_extractor import WavLM_feat


def main():
    parser = argparse.ArgumentParser(description="Export a fine-tuned DeWavLM checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned training checkpoint.")
    parser.add_argument("--output", required=True, help="Output DeWavLM checkpoint path.")
    args = parser.parse_args()

    model = WavLM_feat()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.wavlm.load_state_dict(checkpoint["model"])

    state_dict = {
        "cfg": model.wavlm.cfg.__dict__,
        "model": model.wavlm.state_dict(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output)
    print("Successfully created a pre-trained WavLM checkpoint:", output)


if __name__ == "__main__":
    main()
