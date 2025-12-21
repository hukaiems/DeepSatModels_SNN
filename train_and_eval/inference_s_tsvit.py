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



import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

# --- 1. Define the Evaluation Function ---
def compute_and_plot_cm(model, val_loader, device, num_classes=20, class_names=None, save_path="confusion_matrix.png"):
    """
    Runs inference, computes the Confusion Matrix batch-wise (saves RAM),
    and plots the normalized heatmap (Recall).
    """
    model.eval()
    
    # Initialize empty matrix
    total_cm = np.zeros((num_classes, num_classes))
    
    print("🔍 Starting Confusion Matrix Calculation...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inferencing"):
            # 1. Unpack Batch (Match this to your specific keys)
            inputs = batch['img'].to(device)
            dates = batch['doy'].to(device)
            targets = batch['labels'].to(device)
            
            # 2. Forward Pass
            outputs = model(inputs, dates)
            
            # 3. Get Predictions (Argmax)
            preds = torch.argmax(outputs, dim=1) # Shape: (B, H, W)
            
            # 4. Flatten for Scikit-Learn (B*H*W)
            # Important: Move to CPU immediately to free GPU memory
            preds_flat = preds.flatten().cpu().numpy()
            targets_flat = targets.flatten().cpu().numpy()
            
            # 5. Compute Batch CM
            # 'labels' ensures we track all classes even if missing in this batch
            batch_cm = confusion_matrix(targets_flat, preds_flat, labels=np.arange(num_classes))
            total_cm += batch_cm

    # --- Normalization (Row-wise = Recall) ---
    # Divide by the sum of True Labels (Rows)
    # +1e-7 prevents division by zero for empty classes
    row_sums = total_cm.sum(axis=1)[:, np.newaxis] + 1e-7
    cm_normalized = total_cm.astype('float') / row_sums

    # --- Plotting ---
    plt.figure(figsize=(20, 16))
    
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
        
    sns.heatmap(
        cm_normalized, 
        annot=True,         # Show numbers
        fmt=".2f",          # 2 decimal places
        cmap="Blues",       # Color scheme
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Recall (Sensitivity)'}
    )
    
    plt.ylabel('True Class (Ground Truth)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.title(f'Normalized Confusion Matrix (Total Pixels: {int(total_cm.sum())})', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Confusion Matrix saved to {save_path}")
    plt.show()
    
    return total_cm


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


    # Define Class Names
    PASTIS_CLASSES = [
        "0: Bg/Other", "1: Corn", "2: Meadow", "3: W.Wheat", "4: W.Barley", 
        "5: W.Rapeseed", "6: Spring Barley", "7: Sunflower", "8: Sugar Beet", 
        "9: Water", "10: Forest", "11: Alfalfa", "12: Soybean", "13: Other Cereal", 
        "14: Potatoes", "15: Sorghum", "16: Peas", "17: Triticale", 
        "18: Durum Wheat", "19: Perm. Grass"
    ]

    # CALL THE FUNCTION
    # Assuming 'model', 'val_loader', and 'device' are already defined in your script
    cm = compute_and_plot_cm(
        model=model, 
        val_loader=val_loader, 
        device=device, 
        num_classes=20, 
        class_names=PASTIS_CLASSES
    )

    # ---------------------------------------------------------
    # ⚡ PHASE 1: Energy Efficiency Analysis
    # ---------------------------------------------------------
    # This runs BEFORE the accuracy loop. It pushes data through,
    # counts spikes, and prints the "15x Efficiency" stat.
    # ---------------------------------------------------------
    measure_energy_efficiency_full(model, val_loader, device, disable_tqdm=args.no_progress_bar)

    # ---------------------------------------------------------
    # 🎯 PHASE 2: Accuracy Evaluation (mIoU)
    # ---------------------------------------------------------
    print("\n--- 🎯 Starting Accuracy Evaluation ---")
    metric = MulticlassJaccardIndex(num_classes=20, average='macro').to(device)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Accuracy", disable=args.no_progress_bar):
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
