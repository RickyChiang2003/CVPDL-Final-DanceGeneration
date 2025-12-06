import torch
import numpy as np
from unimumo.models import UniMuMo
import os
import shutil
import glob
from tqdm import tqdm
from unimumo.motion.utils import visualize_music_motion
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="UniMuMo Motion Retrieval")

    parser.add_argument("--skeleton_dir", type=str, required=True,
                        help="Path to skeleton directory containing .npy motion files")

    parser.add_argument("--query_file", type=str, required=True,
                        help="Path to the query .npy motion file")

    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to UniMuMo checkpoint (full.ckpt)")

    parser.add_argument("--save_dir", type=str, default="./visualize",
                        help="Directory to save visualization results")

    return parser.parse_args()

def tokens_to_embedding(model, token_ids: torch.Tensor, pool: str = 'mean', normalize: bool = True) -> torch.Tensor:
    device = token_ids.device
    embeddings = model.motion_vqvae.quantizer.decode(token_ids.to(device))  # [B, D, T]
    embeddings = embeddings.permute(0, 2, 1)

    if pool == 'mean':
        pooled = embeddings.mean(dim=1)
    elif pool == 'max':
        pooled, _ = embeddings.max(dim=1)
    else:
        raise ValueError(f"Unknown pool type: {pool}")

    if normalize:
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)

    return pooled

def main():
    args = get_args()

    skeleton_dir = args.skeleton_dir
    query_file = args.query_file
    ckpt_path = args.ckpt_path
    save_dir = args.save_dir

    if os.path.exists(save_dir):
        for item in os.listdir(save_dir):
            path = os.path.join(save_dir, item)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    else:
        os.makedirs(save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UniMuMo.from_checkpoint(ckpt_path, device=device, debug=False)
    model.eval()

    pattern = os.path.join(skeleton_dir, "**", "*.npy")
    skeleton_files = sorted(glob.glob(pattern, recursive=True))

    all_embeddings = []
    file_names = []

    print("Encoding all skeleton motion files...")
    for file_path in tqdm(skeleton_files):
        skeleton = np.load(file_path)
        skeleton_tensor = torch.FloatTensor(skeleton)[None, :, :]

        try:
            skeleton_latent = model.encode_motion(skeleton_tensor).long()
            emb = tokens_to_embedding(model, skeleton_latent).cpu()
        except RuntimeError as e:
            print(f"Skipping {file_path} due to error: {e}")
            continue

        all_embeddings.append(emb)
        file_names.append(file_path)

    print(f"\nEncoding query file: {query_file}")
    skeleton = np.load(query_file)
    skeleton_tensor = torch.FloatTensor(skeleton)[None, :, :]

    try:
        skeleton_latent = model.encode_motion(skeleton_tensor).long()
        query_emb = tokens_to_embedding(model, skeleton_latent).cpu()
    except RuntimeError as e:
        print(f"Query motion error: {e}")
        return

    print("\nComputing similarity...")
    query_emb_mean = query_emb.mean(axis=0, keepdims=True)

    scores = []
    for emb in all_embeddings:
        score = cosine_similarity(query_emb_mean, emb.mean(axis=0, keepdims=True))[0][0]
        scores.append(score)

    top10_idx = np.argsort(scores)[-10:][::-1]

    output_path = os.path.join(save_dir, "result.txt")
    vis_files = []

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Top 10 most similar files:\n")
        for rank, idx in enumerate(top10_idx, 1):
            line = f"{rank}. {file_names[idx]} - Cosine Similarity: {scores[idx]:.4f}\n"
            print(line, end="")
            f.write(line)
            vis_files.append(file_names[idx])

    print("\nVisualizing motions...")
    for i, filename in enumerate(vis_files):
        motion = np.load(filename)
        waveform = np.zeros((motion.shape[0], 1, motion.shape[1] * 160))

        motion_to_visualize = model.motion_vec_to_joint(
            torch.Tensor(model.normalize_motion(motion))
        )
        motion_to_visualize = np.expand_dims(motion_to_visualize, axis=0)

        visualize_music_motion(
            waveform=waveform,
            joint=motion_to_visualize,
            save_dir=os.path.join(save_dir, str(i)),
            fps=20
        )

    print("\nDONE!")


if __name__ == "__main__":
    main()
