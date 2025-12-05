# UniMuMo: House Dance Inpainting Project

This project adapts the UniMuMo model to perform music-driven motion inpainting on House dance data derived from the AIST++ dataset.
The specific task is to take a 24-beat music segment and a 24-beat motion sequence where the middle 8 beats are masked, and generate the missing middle motion conditioned on the music and the surrounding motion context.

## 1. Environment Setup

Create and activate the `unimumo` conda environment (using Python 3.9.25):

```bash
conda create -n unimumo python=3.9.25
conda activate unimumo
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Project Structure

- `UniMuMo/`: Main codebase.
  - `train.py`: Main training script.
  - `inference_house.py`: Script for inference and visualization.
  - `configs/train_processed_house_scratch.yaml`: Configuration for training from scratch.
  - `unimumo/data/processed_house_dataset.py`: Dataset loader for the 24-beat task.
- `data/`: Data directory.
  - `processed_house/`: Contains processed motion and music data.
- `training_logs/`: Stores training logs and checkpoints.
- `inference_results/`: Stores output videos from inference.

## 3. Data Preparation

The motion data used in this project is derived from the **AIST++** dataset (https://google.github.io/aistplusplus_dataset/factsfigures.html). We have already processed and converted the data for this task.

1. Download the processed AIST++ data (`AIST++.zip`) and the pre-trained model (`best.ckpt`) from Google Drive:
   - Link: https://drive.google.com/drive/u/2/folders/1VtziBnQZqa88lQ42RO-AV-Upc2J39Bjz
   - You can use `gdown` or manually download it.

2. Unzip the data file and place it in the `data/` directory:

```bash
unzip AIST++.zip -d data/
```

Ensure the directory structure looks like `data/AIST++/...`.

3. Place the `best.ckpt` file in the `UniMuMo/` directory (or any preferred location).

**Note:** Since the data is already processed, you **do not** need to run any preprocessing scripts (like `preprocess_house.py`).

## 4. Usage

### Step 1: Training

To train the model from scratch (fine-tuning the LM on the new dataset) using the provided configuration:

```bash
cd UniMuMo
python train.py -b configs/train_processed_house_scratch.yaml --stage train_motion_inpaint
```

**Training Notes:**
- The training configuration is set to run indefinitely (`max_epochs: -1`).
- You should monitor the training progress using TensorBoard to decide when to stop. Look for the point where `val/loss` stops decreasing or starts to increase (indicating overfitting).
  ```bash
  tensorboard --logdir training_logs
  ```
- To stop training, manually interrupt the process (Ctrl+C).
- To resume from a checkpoint, add `-r training_logs/YOUR_LOG_DIR`.

### Step 2: Inference

To generate results using a trained checkpoint:

```bash
cd UniMuMo
python inference_house.py --ckpt best.ckpt
```

(Or point to your training checkpoint: `training_logs/YOUR_LOG_DIR/checkpoints/last.ckpt`)

**Options:**
- `--input_index N`: Specify the index of the sample to use from the **test set** (in `data/processed_house/processed_codes/test`). The index corresponds to the sorted file list. If omitted, a random sample is used.
- `--save_path DIR`: Directory to save the output video (default: `inference_results`).
- `--data_dir DIR`: Path to data directory (default: `../data/processed_house/processed_codes`).

The output will be a video `final_comparison.mp4` showing:
- **Left**: Ground Truth.
- **Right**: Prediction (Middle 8 beats colored in orange).

### Step 3: Evaluation

To compute quantitative metrics (Kinetic/Geometric Distribution Spread, Beat Alignment Score, Kinetic Diversity, FID) on the entire test set:

```bash
cd UniMuMo
python evaluate_house.py --ckpt best.ckpt
```

**Options:**
- `--data_dir DIR`: Path to test data (default: `../data/processed_house/processed_codes/test`).
- `--save_path DIR`: Directory to save the metrics report (default: `evaluation_results`).

The script will output the calculated metrics to the console and save them to `evaluation_results/metrics.txt`.
