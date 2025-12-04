import sys
import os

# 1. Get the directory of the current script (train_stsvit.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (DeepSatModels_SNN)
# This is where 'data' and 'models' live
repo_root = os.path.dirname(current_dir)

# 3. Add the repo root to the Python path
if repo_root not in sys.path:
    sys.path.append(repo_root)
    print(f"🔗 Added repo root to path: {repo_root}")


import argparse # for adding argument in the cmd line
import torch
import torch.nn as nn # nn = neural network
import torch.optim as optim  # optimizer lib like adam, sgd, ...
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from torchmetrics.classification import MulticlassJaccardIndex # mIoU score

from spike_data.pastis_dataset import PastisDataset
from models.snn.spike_tsvit import SpikeTSViTMean, SpikeTSViTNoMean
from spikingjelly.clock_driven.functional import reset_net
from models.snn.helper_functions import measure_energy_efficiency_full

# --- ARGUMENT PARSER ---
def get_args():
    parser = argparse.ArgumentParser(description="Train SpikeTSViT on PASTIS")

    # Paths
    parser.add_argument('--val_csv_path', type=str, default=None, help="Optional: Path to Validation CSV. If None, splits csv_path.")
    parser.add_argument('--data_root', type=str, default="/kaggle/input", help="Root dir of dataset")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to .pth checkpoint") 

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")

    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=64, help="Embedding dim")
    parser.add_argument('--heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--spatial_depth', type=int, default=1, help='Number of spatial blocks')
    parser.add_argument('--temporal_depth', type=int, default=1, help='Number of temporal blocks')
    parser.add_argument('--max_seq_len', type=int, default=10,
                        help="Fixed time length for input sequences - def=10")
    parser.add_argument('--model_type', type=str, default="mean", choices=['mean', 'no_mean'], help="The architecture type")
    parser.add_argument('--norm_type', type=str, default='gn', 
                        choices=['bn', 'gn'],
                        help="Normalization layer: 'bn' (Batch Norm) or 'gn' (Group Norm / Layer Norm)")

    # Attention Mode
    parser.add_argument('--att_mode', type=str, default='2D_dot', choices=['2D_dot', '2D_ham'], 
                        help="Attention mechanism: 'dot' (Standard) or 'hamming' (Efficiency)")

    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of CPU processors to load data for the model")
    parser.add_argument('--no_progress_bar', action='store_true', 
                        help="Disable tqdm progress bar (useful for Kaggle Commit/Save Version)")

    return parser.parse_args()


# --- 3. MAIN EXECUTION ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- 🚀 Starting Training ---")
    print(f"📋 EXPERIMENT CONFIGURATION")
    print(f"{'='*40}")
    
    # Paths
    print(f"📂 Paths:")
    print(f"   Data Root:       {args.data_root}")
    print(f"   Checkpoint Path: {args.checkpoint_path}")

    # Hyperparameters
    print(f"\n⚙️  Hyperparameters:")
    print(f"   Batch Size:      {args.batch_size}")

    # Model Architecture
    print(f"\n🧠 Model Architecture:")

    if args.model_type == 'mean':
        ModelType = SpikeTSViTMean
        print(f"🧠 Model Selected: SpikeTSViTMean (Using Global Average Pooling)")
    else:
        ModelType = SpikeTSViTNoMean
        print(f"🧠 Model Selected: SpikeTSViTNoMean (Full Temporal Processing)")

    # Norm_type
    if args.norm_type == 'bn':
        print(f"🧠 Normalization type: BatchNorm.")
    else:
        print(f"🧠 Normalization type: GroupNorm.")

    print(f"   Embedding Dim:   {args.embed_dim}")
    print(f"   Attention Heads: {args.heads}")
    print(f"   Temporal Depth:  {args.temporal_depth}")
    print(f"   Spatial Depth:   {args.spatial_depth}")
    print(f"   Max Seq Len:     {args.max_seq_len}")
    print(f"   Attention mode:  {args.att_mode}")

    # Misc
    print(f"\n🔧 System/Misc:")
    print(f"   Num Workers:     {args.num_workers}")
    print(f"{'='*40}\n")


    print(f"   Val:   {args.val_csv_path}")

    val_df = pd.read_csv(args.val_csv_path, header=None)

    val_loader = DataLoader(
        PastisDataset(val_df, args.data_root, max_seq_len=args.max_seq_len, mode='eval'),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True
    )

    # 2. Model Setup
    model = ModelType(
        in_channels=10,
        embed_dim=args.embed_dim,
        num_classes=20,
        spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth,
        att_mode=args.att_mode,
        norm_type=args.norm_type,
        num_heads=args.heads,
    ).to(device)
    
    print(f"🔄 Loading Weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Handle dictionary vs direct state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_score = checkpoint.get('best_score', 'N/A')
        print(f"✅ Loaded Checkpoint (Prev Best mIoU: {best_score})")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded Weights (Direct State Dict)")

    # CRITICAL: Freeze model for testing
    model.eval()

    # ---------------------------------------------------------
    # ⚡ PHASE 1: Energy Efficiency Analysis
    # ---------------------------------------------------------
    # This runs BEFORE the accuracy loop. It pushes data through,
    # counts spikes, and prints the "15x Efficiency" stat.
    # ---------------------------------------------------------
    measure_energy_efficiency_full(model, test_loader, device)

    # ---------------------------------------------------------
    # 🎯 PHASE 2: Accuracy Evaluation (mIoU)
    # ---------------------------------------------------------
    print("\n--- 🎯 Starting Accuracy Evaluation ---")
    metric = MulticlassJaccardIndex(num_classes=20, average='macro').to(device)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Accuracy", disable=args.no_progress_bar):
            x = batch['sequence'].to(device)
            dates = batch['dates'].to(device)
            y = batch['labels'].to(device)

            # Inference
            logits = model(x, dates)
            preds = torch.argmax(logits, dim=1)
            
            # Update Metric
            metric.update(preds, y)
            
            # Reset SNN states (Voltage = 0)
            reset_net(model)

    final_miou = metric.compute().item()
    print(f"\n=========================================")
    print(f"🏆 Final Test mIoU: {final_miou:.4f}")
    print(f"=========================================\n")

if __name__ == "__main__": # only run if execute python command.
    main() 
