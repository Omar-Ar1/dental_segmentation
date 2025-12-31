import argparse
import os

def get_parser():
    """
    Defines arguments common to BOTH training and inference.
    These include data paths, model architecture, and system settings.
    """
    parser = argparse.ArgumentParser(add_help=False) # base parser

    # --- Project / System (Cluster Fixes) ---
    parser.add_argument("--work_dir", type=str, default="/gpfs/workdir/arbiom",
                        help="Base directory for cache, logs, and temp files (avoids Home quota issues)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # --- Data / IO ---
    parser.add_argument("--data_root", type=str, default="dataset/images",
                        help="Path to the directory containing input images")
    parser.add_argument("--json_path", type=str, default="dataset/annotations.json",
                        help="Path to the COCO-style annotation JSON file")
    parser.add_argument("--input_size", type=int, default=518,
                        help="Input image resolution (e.g. 518 for DINOv2/v3)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Fraction of data to use for validation")
    parser.add_argument("--test_frac", type=float, default=0.15,
                        help="Fraction of data to use for testing")

    # --- Model Architecture ---
    parser.add_argument("--backbone", type=str, default="vit_large_patch14_dinov2",
                        help="Name of the timm backbone (e.g., vit_large_patch14_dinov2, vit_giant_patch14_dinov2)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of segmentation classes (including background)")
    parser.add_argument("--num_register_tokens", type=int, default=0,
                        help="Number of register tokens to drop (specific to DINOv3)")
    parser.add_argument("--feat_channels", type=int, default=256,
                        help="Number of feature channels in the FPN decoder")
    parser.add_argument("--taps", type=int, nargs='+', default=[6, 12, 18, 23],)
    
    # --- Loss Functions ---
    parser.add_argument("--use_boundary_loss", action="store_true",
                        help="Enable boundary-aware loss during training")
    parser.add_argument("--lambda_b", type=float, default=0.2,
                        help="Weight for the boundary loss term")

    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=4,
                        help="Gradient accumulation steps")
    
    # --- Checkpointing ---
    parser.add_argument("--save_dir", type=str, default="dental_segmentation/experiments",
                        help="Directory to save logs and checkpoints relative to work_dir")
    parser.add_argument("--tensorboard_logs_dir", type=str, default="dental_segmentation/tensorboard_logs")
    return parser


def get_cfg(args_list=None):
    """
    Helper to parse arguments generically.
    Wraps the base parser in a new ArgumentParser to ensure --help works correctly.
    """
    base_parser = get_parser()
    # Create a new parser that inherits from base_parser
    parser = argparse.ArgumentParser(parents=[base_parser], description="Dental Segmentation Config")
    
    args = parser.parse_args(args_list)
    return args