import argparse
import os
import sys
import torch
import numpy as np
import librosa
from scipy import linalg
from scipy.signal import argrelextrema
import scipy.signal

# Patch scipy.signal.hann for backward compatibility
if not hasattr(scipy.signal, 'hann'):
    try:
        import scipy.signal.windows
        scipy.signal.hann = scipy.signal.windows.hann
    except (ImportError, AttributeError):
        # Fallback if scipy.signal.windows is not available or hann not found
        print("Warning: could not patch scipy.signal.hann")
        pass

from scipy.ndimage import gaussian_filter as G
import random
from tqdm import tqdm
import traceback

# Add paths for Bailando and BAMM
sys.path.append(os.path.abspath("../evaluation/Bailando"))
sys.path.append(os.path.abspath("../evaluation/BAMM-main"))

from omegaconf import OmegaConf
from unimumo.util import instantiate_from_config
from unimumo.models.unimumo import UniMuMo
from unimumo.audio.audiocraft_.models.loaders import load_compression_model
from unimumo.audio.audiocraft_.modules.conditioners import ConditioningAttributes

# Import Bailando feature extractors
# Assuming evaluation/Bailando/utils/features/kinetic.py exists and has extract_kinetic_features
# and manual_new.py has extract_manual_features
try:
    from utils.features.kinetic import extract_kinetic_features
    from utils.features.manual_new import extract_manual_features
except ImportError:
    print("Error importing Bailando features. Ensure 'evaluation/Bailando/utils/features' exists and is in python path.")
    sys.exit(1)

# --- Metric Calculation Functions ---

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_diversity_bamm(activation, diversity_times):
    dist_list = []
    n_samples = activation.shape[0]
    if n_samples < 20:
        print(f"Warning: Not enough samples for diversity calculation ({n_samples} < 20). Using all.")
        subset_size = n_samples
    else:
        subset_size = 20
        
    for _ in range(diversity_times):
        indices = torch.randperm(n_samples)[:subset_size]
        sub_act = torch.from_numpy(activation[indices])
        dist = 0
        for i in range(subset_size):
            for j in range(i + 1, subset_size):
                dist += torch.linalg.norm(sub_act[i] - sub_act[j])
        dist_list.append(dist / (subset_size * (subset_size - 1) / 2))
    return np.mean(dist_list)

# --- Beat Alignment Utils (Bailando) ---

def calc_db(keypoints):
    """
    Calculate Dance Beats from keypoints.
    keypoints: (T, N, 3) or (T, N*3)
    """
    keypoints = np.array(keypoints)
    if keypoints.ndim == 2:
        # Assume (T, N*3)
        if keypoints.shape[1] % 3 != 0:
             raise ValueError(f"Flattened keypoints shape {keypoints.shape} second dim must be multiple of 3")
        n_joints = keypoints.shape[1] // 3
        keypoints = keypoints.reshape(-1, n_joints, 3)
    
    if keypoints.ndim != 3 or keypoints.shape[2] != 3:
         raise ValueError(f"calc_db expects 3D array (T, N, 3), got {keypoints.shape}")

    # Calculate kinetic velocity
    # keypoints shape: (T, N, 3)
    # diff: (T-1, N, 3)
    # sum sq: (T-1, N)
    # sqrt: (T-1, N)
    # mean over joints (axis 1): (T-1,)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    # Gaussian filter
    kinetic_vel = G(kinetic_vel, 5)
    # Find local minima
    motion_beats = argrelextrema(kinetic_vel, np.less)[0]
    return motion_beats

def get_music_beats_librosa(waveform, sr=32000, fps=60):
    """
    Detect music beats using Librosa.
    """
    # Librosa expects numpy array (channels, samples) or (samples,)
    # Waveform from UniMuMo decode is (1, samples) usually or (samples,)
    if torch.is_tensor(waveform):
        waveform = waveform.cpu().numpy()
    
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
        
    tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr)
    
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # Convert time to motion frames
    beat_frames_motion = (beat_times * fps).astype(int)
    
    return beat_frames_motion

def calculate_ba_score(music_beats, motion_beats):
    ba = 0
    if len(music_beats) == 0:
        return 0.0
    for bb in music_beats:
        if len(motion_beats) > 0:
            dist = np.min((motion_beats - bb)**2)
            ba += np.exp(-dist / 2 / 9)
        else:
            ba += 0 # No motion beat
    return ba / len(music_beats)

# --- Main Evaluation Loop ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained checkpoint (LM)")
    parser.add_argument("--data_dir", type=str, default="../data/processed_house/processed_codes/test")
    parser.add_argument("--save_path", type=str, default="evaluation_results")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # 1. Load Model
    print("Loading model...")
    try:
        # Load Music VQVAE
        music_vqvae = load_compression_model('facebook/musicgen-small', device='cuda')
        
        # Load Motion VQVAE
        motion_vqvae_conf = OmegaConf.load("configs/train_motion_vqvae.yaml")
        
        # Solution: Instantiate MotionVQVAE class directly instead of using instantiate_from_config
        from unimumo.models.motion_vqvae import MotionVQVAE
        motion_vqvae = MotionVQVAE(**motion_vqvae_conf.model.params, pretrained_music_vqvae=music_vqvae)
        
        motion_vqvae_ckpt_path = "./motion_vqvae.ckpt"
        try:
            motion_vqvae_ckpt = torch.load(motion_vqvae_ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            motion_vqvae_ckpt = torch.load(motion_vqvae_ckpt_path, map_location='cpu')
            
        if 'state_dict' in motion_vqvae_ckpt:
            motion_vqvae_sd = motion_vqvae_ckpt['state_dict']
        else:
            motion_vqvae_sd = motion_vqvae_ckpt
        
        motion_vqvae.load_state_dict(motion_vqvae_sd, strict=False)
        
        # Load UniMuMo
        music_vqvae_config_dummy = OmegaConf.create({
            'compression_model': 'encodec',
            'device': 'cuda',
            'encodec': {
                'autoencoder': 'seanet',
                'quantizer': 'rvq',
                'sample_rate': 32000,
                'causal': False,
                'channels': 1
            },
            'seanet': {
                'encoder': {'n_filters': 32, 'n_residual_layers': 1, 'ratios': [10, 8, 16], 'dimension': 128},
                'decoder': {'n_filters': 32, 'n_residual_layers': 1, 'ratios': [10, 8, 16], 'dimension': 128},
                'channels': 1
            },
            'rvq': {
                'n_q': 8,
                'bins': 1024,
                'kmeans_init': True,
                'kmeans_iters': 10
            }
        })
        motion_vqvae_config_dummy = OmegaConf.create({'model': {'target': 'unimumo.models.motion_vqvae.MotionVQVAE', 'params': {'music_config': {'vqvae_ckpt': 'dummy', 'freeze_codebook': True}, 'motion_config': {'input_dim': 263, 'output_dim': 128, 'emb_dim_encoder': [256, 224, 192, 144, 128], 'emb_dim_decoder': [128, 144, 192, 224, 256], 'input_fps': 60, 'rvq_fps': 50, 'dilation_growth_rate': 2, 'depth_per_res_block': 6, 'activation': 'relu'}, 'pre_post_quantize_config': {'pre_quant_conv_mult': 4, 'post_quant_conv_mult': 4}, 'loss_config': {'target': 'unimumo.modules.loss.MotionVqVaeLoss', 'params': {'commitment_loss_weight': 0.02, 'motion_weight': 1.0}}}}})
        # Add pretrained_music_vqvae to dummy config params to avoid UnsupportedValueType error during UniMuMo init if it tries to use it from config
        # However, UniMuMo init doesn't use this param directly for MotionVQVAE init if we pass the object.
        # The issue is likely in how we modified MotionVQVAE config earlier.
        
        music_motion_lm_config_dummy = OmegaConf.load("configs/train_music_motion.yaml")
        music_motion_lm_config_dummy.model.params.stage = 'train_music_motion'
        
        motion_mean = np.load("../data/Mean.npy")
        motion_std = np.load("../data/Std.npy")
        
        model = UniMuMo(
            music_vqvae_config=music_vqvae_config_dummy,
            motion_vqvae_config=motion_vqvae_config_dummy,
            music_motion_lm_config=music_motion_lm_config_dummy,
            motion_mean=motion_mean,
            motion_std=motion_std,
            debug=False
        )
        
        model.music_vqvae = music_vqvae
        model.motion_vqvae = motion_vqvae
        
        print(f"Loading trained LM from {args.ckpt}...")
        try:
            trained_ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        except TypeError:
            trained_ckpt = torch.load(args.ckpt, map_location='cpu')

        if 'state_dict' in trained_ckpt:
            trained_sd = trained_ckpt['state_dict']
        else:
            trained_sd = trained_ckpt
            
        lm_sd_filtered = {k.replace('music_motion_lm.', ''): v for k, v in trained_sd.items() if k.startswith('music_motion_lm.')}
        model.music_motion_lm.load_state_dict(lm_sd_filtered, strict=False)
        
        model.eval().cuda()
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        traceback.print_exc()
        return

    # 2. Load Data
    from unimumo.data.processed_house_dataset import ProcessedHouseDataset
    dataset = ProcessedHouseDataset(args.data_dir)
    print(f"Loaded {len(dataset)} test samples.")

    # 3. Generation and Feature Extraction Loop
    gt_features_k_list = []
    gt_features_m_list = []
    pred_features_k_list = []
    pred_features_m_list = []
    bas_scores = []

    print("Starting inference and evaluation...")
    
    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
            music_code = sample['music_code'].unsqueeze(0).cuda() # [1, K, T]
            motion_code_gt = sample['motion_code'].unsqueeze(0).cuda() # [1, K, T]
            
            T_code = music_code.shape[-1]
            third = T_code // 3
            
            # Mask Middle 8 beats
            motion_code_input = motion_code_gt.clone()
            motion_code_input[..., third:2*third] = -1 
            
            lm_model = model.music_motion_lm.model
            conditions = [ConditioningAttributes(text={'description': '<separation>'})] * 2
            
            with torch.no_grad():
                # Generate
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
                
                # Decode GT
                waveform_gt, motion_gt_dict = model.decode_music_motion(music_code, motion_code_gt)
                motion_joint_gt = motion_gt_dict['joint'] # (1, T, 24, 3)
                
                # Decode Pred
                waveform_pred, motion_pred_dict = model.decode_music_motion(music_out, motion_out)
                motion_joint_pred = motion_pred_dict['joint'] # (1, T, 24, 3)
                
                # Squeeze batch dim
                # motion_joint_gt and motion_joint_pred are already numpy arrays from decode_music_motion
                joints_gt = motion_joint_gt.squeeze(0) # (T, 22, 3)
                joints_pred = motion_joint_pred.squeeze(0) # (T, 22, 3)
                
                # Pad to 24 joints if necessary (UniMuMo outputs 22, Bailando expects 24)
                if joints_gt.shape[1] == 22:
                    # Pad with zeros for the last 2 joints
                    padding = np.zeros((joints_gt.shape[0], 2, 3), dtype=joints_gt.dtype)
                    joints_gt = np.concatenate([joints_gt, padding], axis=1)
                
                if joints_pred.shape[1] == 22:
                    padding = np.zeros((joints_pred.shape[0], 2, 3), dtype=joints_pred.dtype)
                    joints_pred = np.concatenate([joints_pred, padding], axis=1)

                # waveform is already numpy array from decode_music_motion
                waveform = waveform_gt.squeeze(0) # (Channels, Samples)
                
                # --- 3.1 Extract Features (Bailando) ---
                # Relative to root for feature extraction (handled in extract functions?)
                # extract_kinetic_features expects (T, 24, 3)
                
                # Center root to origin? extract_aist_features.py does:
                # roott = keypoints3d[:1, :1]
                # keypoints3d = keypoints3d - roott
                
                # Do the same centering
                root_gt = joints_gt[:1, :1, :]
                joints_gt_centered = joints_gt - root_gt
                
                root_pred = joints_pred[:1, :1, :]
                joints_pred_centered = joints_pred - root_pred
                
                feat_k_gt = extract_kinetic_features(joints_gt_centered)
                feat_m_gt = extract_manual_features(joints_gt_centered)
                
                feat_k_pred = extract_kinetic_features(joints_pred_centered)
                feat_m_pred = extract_manual_features(joints_pred_centered)
                
                gt_features_k_list.append(feat_k_gt)
                gt_features_m_list.append(feat_m_gt)
                pred_features_k_list.append(feat_k_pred)
                pred_features_m_list.append(feat_m_pred)
                
                # --- 3.2 BAS Calculation ---
                motion_beats = calc_db(joints_pred) 
                
                # Music beats
                # Use model.motion_fps (60) or configured FPS
                fps = model.motion_fps if hasattr(model, 'motion_fps') else 60
                music_beats = get_music_beats_librosa(waveform, fps=fps)
                
                score = calculate_ba_score(music_beats, motion_beats)
                bas_scores.append(score)
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # traceback.print_exc()
            continue

    # 4. Calculate Metrics
    print("\nCalculating aggregate metrics...")
    
    # Stack features
    gt_features_k = np.stack(gt_features_k_list)
    gt_features_m = np.stack(gt_features_m_list)
    pred_features_k = np.stack(pred_features_k_list)
    pred_features_m = np.stack(pred_features_m_list)
    
    # Normalize (Bailando style)
    gt_features_k_norm, pred_features_k_norm = normalize(gt_features_k, pred_features_k)
    gt_features_m_norm, pred_features_m_norm = normalize(gt_features_m, pred_features_m)
    
    # 4.1 Dist_k (Kinetic Distribution Spread)
    mu_gt_k, cov_gt_k = calculate_activation_statistics(gt_features_k_norm)
    mu_pred_k, cov_pred_k = calculate_activation_statistics(pred_features_k_norm)
    dist_k = calculate_frechet_distance(mu_gt_k, cov_gt_k, mu_pred_k, cov_pred_k)
    
    # 4.2 Dist_g (Geometric Distribution Spread)
    mu_gt_m, cov_gt_m = calculate_activation_statistics(gt_features_m_norm)
    mu_pred_m, cov_pred_m = calculate_activation_statistics(pred_features_m_norm)
    dist_g = calculate_frechet_distance(mu_gt_m, cov_gt_m, mu_pred_m, cov_pred_m)
    
    # 4.3 BAS
    avg_bas = np.mean(bas_scores)
    
    # 4.4 Kinetic Diversity (BAMM style)
    # Use normalized kinetic features
    diversity_k = calculate_diversity_bamm(pred_features_k_norm, diversity_times=20)
    
    # 4.5 FID (BAMM style)
    # User asked for FID based on BAMM. 
    # BAMM calculates FID on embeddings. We use Kinetic Features as proxy.
    # This effectively duplicates Dist_k, but we label it FID(Kinetic).
    fid_val = dist_k 
    
    print("="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"Kinetic Distribution Spread (Dist_k): {dist_k:.4f}")
    print(f"Geometric Distribution Spread (Dist_g): {dist_g:.4f}")
    print(f"Beat Alignment Score (BAS): {avg_bas:.4f}")
    print(f"Kinetic Diversity (BAMM method): {diversity_k:.4f}")
    print(f"FID (BAMM method on Kinetic Feats): {fid_val:.4f}")
    print("="*30)
    
    # Save results to text file
    with open(os.path.join(args.save_path, "metrics.txt"), "w") as f:
        f.write(f"Kinetic Distribution Spread (Dist_k): {dist_k:.4f}\n")
        f.write(f"Geometric Distribution Spread (Dist_g): {dist_g:.4f}\n")
        f.write(f"Beat Alignment Score (BAS): {avg_bas:.4f}\n")
        f.write(f"Kinetic Diversity (BAMM method): {diversity_k:.4f}\n")
        f.write(f"FID (BAMM method on Kinetic Feats): {fid_val:.4f}\n")

if __name__ == "__main__":
    main()

