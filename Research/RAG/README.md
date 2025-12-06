# UniMuMo Motion Retrieval System

This repository provides a command-line tool for performing **motion-to-motion retrieval** using the UniMuMo model.  
Given a query motion file, the script extracts its latent embedding, compares it against a database of motion embeddings, and retrieves the most similar motion sequences based on **cosine similarity**.  
The top results are also **rendered as visualizations** for qualitative inspection.

---

## Features

- Extract motion embeddings using **UniMuMo's Motion VQ-VAE**
- Compute similarity using **cosine similarity**
- Retrieve **Top-K** most similar motion files
- Render retrieved motions using UniMuMo's built-in visualization utilities

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- tqdm
- UniMuMo

---

## Command-Line Arguments

| Argument | Type | Description |
|---------|------|-------------|
| `--skeleton_dir` | `str` | Directory containing `.npy` motion files in HumanML3D format. Searched recursively. |
| `--query_file` | `str` | Path to a single `.npy` motion file (HumanML3D format) used as the query. |
| `--ckpt_path` | `str` | Path to the UniMuMo checkpoint (e.g., `full.ckpt` download from https://github.com/hanyangclarence/UniMuMo). |
| `--save_dir` | `str` | Output directory for results and visualizations. Default: `./visualize` |

---

##  Usage Example

```bash
python motion_RAG.py \
    --skeleton_dir "AIST++/AIST++/new_joint_vecs" \
    --query_file "query_motion.npy" \
    --ckpt_path "full.ckpt" \
    --save_dir "./visualize"
