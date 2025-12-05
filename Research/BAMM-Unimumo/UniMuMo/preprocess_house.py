import argparse
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import re
import sys
from omegaconf import OmegaConf
import torchaudio.transforms as T

sys.path.append(os.getcwd())

from unimumo.audio.audiocraft_.data.audio import audio_read
from unimumo.audio.audiocraft_.models.loaders import load_compression_model, load_compression_model_ckpt
from unimumo.models.motion_vqvae import MotionVQVAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/tmp2/b11705045/CVPDL/Final-project/data/processed_house")
    parser.add_argument("--output_dir", type=str, default="/tmp2/b11705045/CVPDL/Final-project/data/processed_house/processed_codes")
    parser.add_argument("--ckpt", type=str, default="/tmp2/b11705045/CVPDL/Final-project/motion_vqvae.ckpt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.ckpt}...")
    
    try:
        motion_mean = np.load("/tmp2/b11705045/CVPDL/Final-project/data/AIST++/Mean.npy")
        motion_std = np.load("/tmp2/b11705045/CVPDL/Final-project/data/AIST++/Std.npy")
    except Exception as e:
        print(f"Error loading Mean/Std: {e}")
        return

    try:
        motion_vqvae_conf = OmegaConf.load("configs/train_motion_vqvae.yaml")
    except Exception as e:
        print(f"Error loading motion vqvae config: {e}")
        return

    # Get music config and model
    print("Getting MusicGen config and model...")
    try:
        pkg = load_compression_model_ckpt('facebook/musicgen-small')
        music_cfg = OmegaConf.create(pkg['xp.cfg'])
        
        # Fix missing sample_rate for interpolation
        if 'sample_rate' not in music_cfg:
            music_cfg.sample_rate = 32000
            
        # Load model object
        music_vqvae = load_compression_model('facebook/musicgen-small', device='cuda')
            
        # Wrap for MotionVQVAE
        music_conf_dict = {
            'vqvae_config': music_cfg,
            'freeze_codebook': True,
            'vqvae_ckpt': '' 
        }
    except Exception as e:
        print(f"Error getting music config: {e}")
        return

    print("Instantiating Motion VQVAE...")
    try:
        motion_conf = motion_vqvae_conf.model.params
        
        # Pass loaded music_vqvae directly
        motion_vqvae = MotionVQVAE(**motion_conf, pretrained_music_vqvae=music_vqvae)
        
        ckpt = torch.load(args.ckpt, map_location='cpu')
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        
        # Keys are likely music_encoder..., motion_encoder... (no prefix)
        # But just in case check for prefixes
        prefix = 'motion_vqvae.'
        if any(k.startswith(prefix) for k in sd.keys()):
             sd = {k.replace(prefix, ''): v for k, v in sd.items()}
        
        # Also remove 'model.' prefix if present (LightningModule)
        if any(k.startswith('model.') for k in sd.keys()):
             sd = {k.replace('model.', ''): v for k, v in sd.items()}

        missing, unexpected = motion_vqvae.load_state_dict(sd, strict=False)
        print(f"Loaded Motion VQVAE. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        motion_vqvae.to('cuda').eval()
        
    except Exception as e:
        print(f"Error loading motion vqvae: {e}")
        import traceback
        traceback.print_exc()
        return

    motion_dir = os.path.join(args.data_dir, "motion")
    music_dir = os.path.join(args.data_dir, "music")
    subdirs = sorted([d for d in os.listdir(motion_dir) if os.path.isdir(os.path.join(motion_dir, d))])
    
    print(f"Found {len(subdirs)} subdirectories.")

    for subdir in tqdm(subdirs):
        motion_subdir_path = os.path.join(motion_dir, subdir)
        music_subdir_path = os.path.join(music_dir, subdir)

        if not os.path.exists(music_subdir_path):
            continue

        motion_files = sorted(glob.glob(os.path.join(motion_subdir_path, "*_vecs.npy")))
        
        for motion_file in motion_files:
            basename = os.path.basename(motion_file)
            match = re.match(r"clip_beat(\d+)_to_beat(\d+)_vecs\.npy", basename)
            if not match:
                continue
            
            start_beat = match.group(1)
            end_beat = match.group(2)
            
            music_filename = f"clip_beat{start_beat}_to_beat{end_beat}.mp3"
            music_file = os.path.join(music_subdir_path, music_filename)
            
            if not os.path.exists(music_file):
                continue
            
            # Output file check
            out_name = f"{subdir}_beat{start_beat}_beat{end_beat}.pth"
            out_path = os.path.join(args.output_dir, out_name)
            if os.path.exists(out_path):
                continue 

            try:
                motion_feature = np.load(motion_file) # [T, 263]
            except Exception as e:
                print(f"Error loading motion {motion_file}: {e}")
                continue

            try:
                waveform, sr = audio_read(music_file) 
                if sr != 32000:
                      resampler = T.Resample(sr, 32000)
                      waveform = resampler(waveform)
            except Exception as e:
                print(f"Error loading music {music_file}: {e}")
                continue
            
            try:
                with torch.no_grad():
                    # Encode Motion
                    motion_norm = (motion_feature - motion_mean) / motion_std
                    
                    target_length = (motion_norm.shape[0] // 6) * 6
                    if target_length == 0:
                        continue
                    motion_norm = motion_norm[:target_length]
                    
                    motion_tensor = torch.FloatTensor(motion_norm).to('cuda').unsqueeze(0) # [1, T, 263]
                    
                    empty_waveform = torch.zeros((1, 1, target_length * 32000 // 60)).to('cuda')
                    
                    _, motion_emb = motion_vqvae.encode(x_music=empty_waveform, x_motion=motion_tensor)
                    motion_code = motion_vqvae.quantizer.encode(motion_emb).contiguous() # [1, K, T_code]
                    
                    # Encode Music
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    waveform = waveform.to('cuda').unsqueeze(0) # [1, 1, T]
                    
                    music_emb = motion_vqvae.music_encoder(waveform)
                    music_code = motion_vqvae.quantizer.encode(music_emb) # [1, K, T]
                    
                    save_dict = {
                        "music_code": music_code.cpu(),
                        "motion_code": motion_code.cpu(),
                        "start_beat": int(start_beat),
                        "end_beat": int(end_beat),
                        "subdir": subdir
                    }
                    
                    torch.save(save_dict, out_path)
                    
            except Exception as e:
                print(f"Error encoding {subdir} {start_beat}-{end_beat}: {e}")
                import traceback
                traceback.print_exc()
                continue

if __name__ == "__main__":
    main()
