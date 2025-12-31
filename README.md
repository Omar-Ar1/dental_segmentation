# Dental Segmentation with Pretrained Vision Transformer

A deep learning pipeline for extracting teeth from radiographic images using **DINOv2/v3** Vision Transformer backbones and **PyTorch Lightning**.

## 🚀 Features

- **Advanced Backbones**: Leverages powerful, pre-trained Vision Transformers (DINOv2, DINOv3) via `timm`.
- **Custom Architecture**: Implements a Multi-scale Feature Pyramid Network (FPN) Decoder for high-resolution segmentation.
- **Boundary-Aware**: Includes an optional Boundary Loss to sharpen segmentation edges.
- **Modular Design**: Built with PyTorch Lightning for clean, scalable, and reproducible training loops.
- **Robust Metrics**: Automatically tracks Dice Score and Hausdorff Distance during training and evaluation.

## 📂 Project Structure

```text
dental-segmentation/
├── src/
│   ├── data/           # Dataset loading and Lightning DataModule
│   ├── models/         # DINOv2 Wrapper and Lightning Module logic
│   └── utils/          # Metrics (Dice, Hausdorff) and visualization tools
├── configs/            # Configuration files
├── train.py            # Main entry point for training and testing
├── pyproject.toml      # Project dependencies and metadata
└── README.md

```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/dental-segmentation.git](https://github.com/Omar-Ar1/dental-segmentation.git)
cd dental-segmentation

```


2. **Create a virtual environment (recommended):**
```bash
conda create -n dental python=3.10 -y
conda activate dental

```


3. **Install dependencies:**
```bash
pip install -e .

```



## 🏃 Usage

### 1. Data Preparation

Ensure your data is formatted with a COCO-style JSON annotation file.

* **Images**: Directory containing the X-ray images.
* **Annotations**: JSON file linking images to segmentation masks.

### 2. Training

To start training the model, run the main script. You can modify the configuration dictionary within `train.py` or pass arguments if configured.

```bash
python train.py

```

**Key Hyperparameters:**

* `backbone`: Choose between variants like `vit_large_patch14_dinov2` or `vit_giant_patch14_dinov2`.
* `batch_size`: Adjust according to your GPU memory (default: 4).
* `accum`: Gradient accumulation steps to simulate larger batches.
* `lr`: Learning rate (default: 1e-4).

### 3. Evaluation

The pipeline automatically runs evaluation on the test set after training completes. To manually evaluate a specific checkpoint:

```bash
python train.py --eval_only --save_dir checkpoints/model_best.ckpt

```

## 📊 Metrics

The model monitors the following metrics to ensure segmentation quality:

* **Dice Score**: Measures the overlap between the predicted segmentation and the ground truth.
* **Hausdorff Distance (HD95)**: Measures the maximum distance between the prediction boundary and the ground truth boundary, ensuring shape accuracy.

