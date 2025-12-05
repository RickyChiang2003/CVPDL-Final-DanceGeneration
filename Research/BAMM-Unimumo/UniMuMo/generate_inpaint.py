import argparse
import os
import torch
import numpy as np
import sys
import subprocess
sys.path.append(os.getcwd())

from unimumo.models import UniMuMo
from unimumo.audio.audiocraft_.modules.conditioners import ConditioningAttributes
from unimumo.motion.utils import visualize_music_motion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="../full.ckpt")
    parser.add_argument("--save_path", type=str, default="../infilling_results")
    parser.add_argument("--motion_file", type=str, default="../data/AIST++/new_joint_vecs/gBR_sBM_cAll_d04_mBR0_ch01.npy")
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Loading model from {args.ckpt}...")
    try:
        model = UniMuMo.from_checkpoint(args.ckpt, device='cuda')
    except Exception as e:
        print(f"Failed to load full model: {e}")
        return

    model.eval()
    
    # Load Motion Data (AIST++)
    print(f"Loading motion from {args.motion_file}...")
    if not os.path.exists(args.motion_file):
        print(f"File not found: {args.motion_file}")
        return
        
    motion_feature = np.load(args.motion_file) # [T, 263]
    print(f"Original motion shape: {motion_feature.shape}")

    # Ensure motion length is compatible
    if model.motion_fps == 20:
        target_motion_length = (motion_feature.shape[0] // 2) * 2
    else: # motion_fps == 60
        target_motion_length = (motion_feature.shape[0] // 6) * 6
    motion_feature = motion_feature[:target_motion_length]
    
    # For testing 8+8+8 structure, we might want to crop to a specific length if needed,
    # or just use 1/3 splits of whatever length we have.
    # Let's use the full length divided into 3 parts.
    
    print("Encoding motion to get GT codes...")
    with torch.no_grad():
        # encode_motion expects numpy [T, 263]
        motion_code_gt = model.encode_motion(motion_feature)  # [1, K, T_code]
    
    T_code = motion_code_gt.shape[-1]
    print(f"Motion code length: {T_code}")
    
    # Create dummy music code (silence/placeholder) since AIST++ has no music attached here easily
    # In a real scenario, we would extract audio features.
    # For now, we use zeros, which means "silence" or "unconditional" roughly.
    music_code = torch.zeros_like(motion_code_gt) # [1, K, T_code]
    
    # Prepare Input for Inpainting
    motion_code_input = motion_code_gt.clone()
    third = T_code // 3
    
    # Mask Middle 33%
    motion_code_input[..., third:2*third] = -1
    
    print(f"Masking codes from {third} to {2*third} (Total: {T_code})")
    
    print("Generating...")
    conditions = [ConditioningAttributes(text={'description': '<separation>'})] * 2
    lm_model = model.music_motion_lm.model
    
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
        
    print("Generation done.")
    
    # Decode
    print("Decoding...")
    # GT
    waveform_gt, motion_gt_dict = model.decode_music_motion(music_code, motion_code_gt)
    motion_joint_gt = motion_gt_dict['joint']
    
    # Prediction (Condition + Generated + Condition)
    # Note: motion_out from generate contains the full sequence (filled in)
    waveform_pred, motion_pred_dict = model.decode_music_motion(music_out, motion_out)
    motion_joint_pred = motion_pred_dict['joint']
    
    # Visualization
    # Define Color Function for Orange Middle
    total_frames = motion_joint_pred.shape[1]
    third_frames = total_frames // 3
    
    def get_orange_color(index):
        # If index is in the middle third, return orange colors
        if third_frames <= index < 2 * third_frames:
            return ['orange', 'orange', 'orange', 'orange', 'orange']
        return None # Use default colors
        
    print("Visualizing GT...")
    visualize_music_motion(
        waveform=waveform_gt[None, ...], 
        joint=motion_joint_gt, 
        save_dir=args.save_path, 
        fps=model.motion_fps,
        filename="final_gt"
    )
    
    print("Visualizing Prediction (with orange middle)...")
    visualize_music_motion(
        waveform=waveform_pred[None, ...], 
        joint=motion_joint_pred, 
        save_dir=args.save_path, 
        fps=model.motion_fps,
        filename="final_pred",
        custom_color_func=get_orange_color
    )
    
    # Combine Side-by-Side
    gt_path = os.path.join(args.save_path, "final_gt_0.mp4")
    pred_path = os.path.join(args.save_path, "final_pred_0.mp4")
    output_path = os.path.join(args.save_path, "comparison_result.mp4")
    
    if os.path.exists(gt_path) and os.path.exists(pred_path):
        print("Combining videos...")
        cmd = f"ffmpeg -y -i {gt_path} -i {pred_path} -filter_complex hstack {output_path}"
        subprocess.call(cmd, shell=True)
        print(f"Comparison saved to {output_path}")
    else:
        print("Error: Could not find generated videos to combine.")

if __name__ == "__main__":
    main()
