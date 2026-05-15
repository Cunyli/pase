# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import sys
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from librosa.util import find_files
from models.wavlm.feature_extractor import WavLM_feat as Encoder
from models.vocoder.wavlmdec_dual import WavLMDec as Model
from utils.config_utils import load_config


def load_pair_map(pair_csv):
    if not pair_csv:
        return {}
    pair_csv = os.path.expanduser(os.path.expandvars(str(pair_csv)))
    if not os.path.isfile(pair_csv):
        return {}
    pairs = {}
    with open(pair_csv, newline="") as f:
        for row in csv.DictReader(f):
            noisy_path = os.path.abspath(row["noisy_filepath"])
            pairs[noisy_path] = {
                "uid": row.get("uid") or Path(noisy_path).stem,
                "clean_path": row["clean_filepath"],
            }
    return pairs


@torch.inference_mode()
def infer(args):
    cfg_infer = load_config(args.config)
    cfg_network = load_config(cfg_infer.network.config)
    
    noisy_folder = cfg_infer.test_dataset.noisy_dir
    clean_folder = cfg_infer.test_dataset.clean_dir
    save_folder = cfg_infer.network.enh_folder
    wav_folder = os.path.join(save_folder, "wav")
    os.makedirs(wav_folder, exist_ok=True)
    
    ext = cfg_infer.test_dataset.extension
    pair_map = load_pair_map(cfg_infer.test_dataset.get("pair_csv", None))
    
    wavs = sorted(find_files(noisy_folder, ext=ext))
    print(f"Inference on folder: {noisy_folder}, {len(wavs)} files")
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(**cfg_network['encoder_config']).to(device)
    vocoder_config = dict(cfg_network['vocoder_config'])
    vocoder_config.pop("pretrained_ckpt_path", None)
    model = Model(**vocoder_config).to(device).eval()
    
    model.load_state_dict(
        torch.load(cfg_infer['network']['checkpoint'], map_location=device)['generator']
    )


    inf_scp_list = []
    ref_scp_list = []
    
    for wav_path in tqdm(wavs):
        noisy, fs = sf.read(wav_path, dtype='float32')
            
        input = torch.FloatTensor(noisy)[None,None].to(device)
        
        feat_a, feat_p = encoder(input)
        output  = model(feat_p, feat_a)
        
        esti_wav = output.cpu().detach().numpy().squeeze()
        esti_wav = esti_wav / np.max(np.abs(esti_wav)) * 0.9
        
        if esti_wav.shape[-1] < noisy.shape[-1]:
            esti_wav = np.pad(esti_wav, (0, noisy.shape[-1]-esti_wav.shape[-1]))
        else:
            esti_wav = esti_wav[..., :noisy.shape[-1]]
        
        pair_info = pair_map.get(os.path.abspath(wav_path), {})
        uid = pair_info.get("uid") or os.path.basename(wav_path).split(f'.{ext}')[0]
        true_path = pair_info.get("clean_path") or os.path.join(clean_folder, f'{uid}.{ext}')
        esti_path = os.path.join(wav_folder, f'{uid}.{ext}')
    
        sf.write(esti_path, esti_wav, fs)
        
        inf_scp_list.append([uid, esti_path])
        ref_scp_list.append([uid, true_path])
        
    # Save paths into scp file for evaluation
    with open(os.path.join(save_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    with open(os.path.join(save_folder, "ref.scp"), "w") as f:
        for uid, audio_path in ref_scp_list:
            f.write(f"{uid} {audio_path}\n")

            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/infer/vocoder_dual_tau_fixed.yaml')
    parser.add_argument('-D', '--device', default='0', help='Index of the gpu device')

    args = parser.parse_args()
    infer(args)
