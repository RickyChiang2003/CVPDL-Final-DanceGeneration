import argparse
import os
import torch
import numpy as np
import sys
import subprocess
import glob
import random
from omegaconf import OmegaConf

sys.path.append(os.getcwd())

from unimumo.models import UniMuMo
from unimumo.audio.audiocraft_.modules.conditioners import ConditioningAttributes
from unimumo.motion.utils import visualize_music_motion
from unimumo.audio.audiocraft_.models.loaders import load_compression_model, load_compression_model_ckpt
from unimumo.models.motion_vqvae import MotionVQVAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained checkpoint (LM)")
    parser.add_argument("--save_path", type=str, default="inference_results")
    parser.add_argument("--data_dir", type=str, default="../data/processed_house/processed_codes")
    parser.add_argument("--input_index", type=int, default=None, help="Index of the sample to infer (optional)")
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Loading model...")
    try:
        # 1. Load Configs
        motion_vqvae_conf = OmegaConf.load("configs/train_motion_vqvae.yaml")
        
        # Get Music Config & Model
        pkg = load_compression_model_ckpt('facebook/musicgen-small')
        music_cfg = OmegaConf.create(pkg['xp.cfg'])
        if 'sample_rate' not in music_cfg:
            music_cfg.sample_rate = 32000
            
        music_vqvae = load_compression_model('facebook/musicgen-small', device='cuda')
        
        lm_conf = OmegaConf.load("configs/train_music_motion.yaml")
        lm_conf.model.params.stage = 'train_music_motion' # Default stage for loading
        
        # Motion Mean/Std
        motion_mean = np.load("../data/AIST++/Mean.npy")
        motion_std = np.load("../data/AIST++/Std.npy")
        
        # Instantiate UniMuMo
        # We pass the loaded music_vqvae to avoid reloading/config issues
        # But UniMuMo __init__ expects config. 
        # We can modify UniMuMo class or just use the same trick:
        # But wait, UniMuMo class does `self.music_vqvae = get_compression_model(...)`.
        # We cannot easily inject the object unless we modify UniMuMo.
        # However, for inference, we can instantiate it and then OVERWRITE self.music_vqvae.
        
        # music_vqvae_config must be valid.
        music_vqvae_config = music_cfg
        motion_vqvae_config = motion_vqvae_conf
        music_motion_lm_config = lm_conf
        
        model = UniMuMo(
            music_vqvae_config=music_vqvae_config,
            motion_vqvae_config=motion_vqvae_config,
            music_motion_lm_config=music_motion_lm_config,
            motion_mean=motion_mean,
            motion_std=motion_std,
            debug=False
        )
        
        # OVERWRITE music_vqvae with the pre-loaded one (which has weights)
        model.music_vqvae = music_vqvae
        
        # Load Motion VQVAE weights
        print("Loading Motion VQVAE weights...")
        base_ckpt_path = "/tmp2/b11705045/CVPDL/Final-project/motion_vqvae.ckpt"
        try:
            base_ckpt = torch.load(base_ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            base_ckpt = torch.load(base_ckpt_path, map_location='cpu')

        if 'state_dict' in base_ckpt:
            base_sd = base_ckpt['state_dict']
        else:
            base_sd = base_ckpt
            
        # Filter keys for MotionVQVAE
        # model.motion_vqvae keys in UniMuMo match keys in base_sd directly (music_encoder..., motion_encoder...)
        # But we need to be careful about `music_encoder` in MotionVQVAE.
        # It is initialized in MotionVQVAE using `instantiate_music_vqvae`.
        # Loading state dict should update it.
        model.motion_vqvae.load_state_dict(base_sd, strict=False)
        
        # Ensure MotionVQVAE uses the same quantizer/encoder as music_vqvae if possible?
        # Or just trust the weights.
        
        # Load LM weights
        print(f"Loading trained LM from {args.ckpt}...")
        try:
            trained_ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        except TypeError:
            trained_ckpt = torch.load(args.ckpt, map_location='cpu')

        if 'state_dict' in trained_ckpt:
            trained_sd = trained_ckpt['state_dict']
        else:
            trained_sd = trained_ckpt
            
        # Load directly into MusicMotionTransformer
        # The keys in trained_sd (model.xxx) match MusicMotionTransformer structure (self.model.xxx)
        missing, unexpected = model.music_motion_lm.load_state_dict(trained_sd, strict=False)
        print(f"Loaded LM. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        model.eval().cuda()
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load Data
    from unimumo.data.processed_house_dataset import ProcessedHouseDataset
    dataset = ProcessedHouseDataset(args.data_dir)
    if len(dataset) == 0:
        print("Dataset empty.")
        return
        
    if args.input_index is not None:
        if 0 <= args.input_index < len(dataset):
            idx = args.input_index
            print(f"Using specified sample index: {idx}")
        else:
            print(f"Error: Index {args.input_index} out of range (0-{len(dataset)-1}). Using random.")
            idx = random.randint(0, len(dataset)-1)
    else:
        idx = random.randint(0, len(dataset)-1)
        print(f"Using random sample index: {idx}")
        
    sample = dataset[idx]
    
    print(f"Sample files:\n  1: {os.path.basename(dataset.samples[idx]['file1'])}\n  2: {os.path.basename(dataset.samples[idx]['file2'])}\n  3: {os.path.basename(dataset.samples[idx]['file3'])}")
    
    music_code = sample['music_code'].unsqueeze(0).cuda() # [1, K, T]
    motion_code_gt = sample['motion_code'].unsqueeze(0).cuda() # [1, K, T]
    
    print(f"Sample loaded. Length: {music_code.shape[-1]}")
    
    # Mask Middle 8 beats (approx 1/3)
    T_code = music_code.shape[-1]
    third = T_code // 3
    
    motion_code_input = motion_code_gt.clone()
    motion_code_input[..., third:2*third] = -1 
    
    # Generate
    print("Generating...")
    lm_model = model.music_motion_lm.model # Access inner LMModel
    # LMModel expects conditions list of length 2*B (music_conds + motion_conds)
    conditions = [ConditioningAttributes(text={'description': '<separation>'})] * 2 # Batch size 1, so 2 conditions
    
    with torch.no_grad():
        music_out, motion_out = lm_model.generate(
            conditions=conditions,
            mode='motion_inpaint',
            music_code=music_code,
            motion_code=motion_code_input, 
            max_gen_len=T_code,
            use_sampling=True,
            temp=1.0,
            top_k=250,
            cfg_coef=4.0
        )
    
    print("Decoding...")
    # Decode GT
    waveform_gt, motion_gt_dict = model.decode_music_motion(music_code, motion_code_gt)
    motion_joint_gt = motion_gt_dict['joint']
    
    # Decode Pred
    waveform_pred, motion_pred_dict = model.decode_music_motion(music_out, motion_out)
    motion_joint_pred = motion_pred_dict['joint']
    
    # Visualization
    fps = model.motion_fps
    total_frames = motion_joint_pred.shape[1]
    
    frame_third = total_frames // 3
    
    def get_orange_color(index):
        if frame_third <= index < 2 * frame_third:
            return ['orange'] * 5 
        return None
    
    print("Visualizing GT...")
    visualize_music_motion(
        waveform=waveform_gt[None, ...],
        joint=motion_joint_gt,
        save_dir=args.save_path,
        fps=fps,
        filename="gt"
    )
    
    print("Visualizing Prediction...")
    visualize_music_motion(
        waveform=waveform_pred[None, ...],
        joint=motion_joint_pred,
        save_dir=args.save_path,
        fps=fps,
        filename="pred",
        custom_color_func=get_orange_color
    )
    
    # Combine
    gt_path = os.path.join(args.save_path, "gt_0.mp4")
    pred_path = os.path.join(args.save_path, "pred_0.mp4")
    output_path = os.path.join(args.save_path, "final_comparison.mp4")
    
    if os.path.exists(gt_path) and os.path.exists(pred_path):
        cmd = f"ffmpeg -y -i {gt_path} -i {pred_path} -filter_complex hstack {output_path}"
        subprocess.call(cmd, shell=True)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
