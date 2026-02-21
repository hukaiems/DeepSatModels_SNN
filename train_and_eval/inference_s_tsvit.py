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
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy # mIoU and OA score

from spike_data.pastis_dataset import PastisDataset, PASTIS_CLASSES
from spike_data.france_dataset import FranceDataset, FRANCE_CLASSES
from models.snn.spike_tsvit import SpikeTSViTMean, SpikeTSViTNoMean
from spikingjelly.clock_driven.functional import reset_net
from models.snn.helper_functions import measure_energy_efficiency_full, check_class_imbalance, compute_and_plot_cm, plot_phenological_confusion, plot_segmentation_comparison, analyze_temporal_importance, visualize_cloud_sensitivity

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
    parser.add_argument('--datasets', type=str, default='pastis', choices=['pastis', 'france'], help='choose the dataset')

    # Attention Mode
    parser.add_argument('--att_mode', type=str, default='2D_dot', choices=['2D_dot', '2D_ham'], 
                        help="Attention mechanism: 'dot' (Standard) or 'hamming' (Efficiency)")

    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of CPU processors to load data for the model")
    # Testing type
    parser.add_argument('--no_progress_bar', action='store_true', 
                        help="Disable tqdm progress bar (useful for Kaggle Commit/Save Version)")
    parser.add_argument('--inference', action='store_true',
                        help='Run inferencing for the checkpoint')
    parser.add_argument('--test_per_class', action='store_true',
                        help="Test the per class mIoU")
    parser.add_argument('--test_energy', action='store_true',
                        help="Test the energy efficiency of the model")

    parser.add_argument('--confusion_matrix', action='store_true',
                        help="Run confusion matrix of the checkpoint")
    parser.add_argument('--NDVI', action='store_true',
                        help="Run NDVI to check similar growth cycle for specific crop types")
    parser.add_argument('--visual_comparison', action='store_true',
                        help="Visualize the error map")
    parser.add_argument('--temporal_importance', action='store_true',
                        help="Plotting out which time step is the most importance for the accuracy of each class.")
    parser.add_argument('--temporal_importance_class', type=int, default=9,
                        help="This is the class number you want to plot the temporal importance experiment.")
    parser.add_argument('--analyze_cloud', action='store_true',
                        help=" Analyzing the cloud cover picture and plot out the prediction")
    
    parser.add_argument('--deploy_inference', action='store_true',
                        help="This is to run deployment inferencing!")

    return parser.parse_args()


# --- 3. MAIN EXECUTION ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- 🚀 Starting Inferencing ---")
    print(f"📋 EXPERIMENT CONFIGURATION")
    print(f"{'='*40}")
    
    # Paths
    print(f"📂 Paths:")
    print(f"   Data Root:       {args.data_root}")
    print(f"   Checkpoint Path: {args.checkpoint_path}")

    # Hyperparameters
    print(f"\n⚙️  Hyperparameters:")
    print(f"   Batch Size:      {args.batch_size}")

    if args.datasets == 'france':
        DatasetLoader = FranceDataset
        in_channels = 13
        num_classes = 21
        ignore_index = 20
        class_names=FRANCE_CLASSES
        print(f"Dataset Selected: France dataset.")
    else: 
        DatasetLoader = PastisDataset
        in_channels = 10
        num_classes = 20
        ignore_index = 19
        class_names=PASTIS_CLASSES
        print(f"Dataset Selected: PASTIS dataset.")

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
        DatasetLoader(val_df, args.data_root, max_seq_len=args.max_seq_len, mode='eval'),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True
    )

    # 2. Model Setup
    model = ModelType(
        in_channels=in_channels,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
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
    # Plot cloud analysis
    # ---------------------------------------------------------
    if args.analyze_cloud:
        visualize_cloud_sensitivity(model, val_loader, device, datasets=args.datasets)

    # ---------------------------------------------------------
    # Plot temporal importance for a specific class
    # ---------------------------------------------------------
    if args.temporal_importance:
        analyze_temporal_importance(model, val_loader, device, target_class=args.temporal_importance_class)


    # ---------------------------------------------------------
    # Plot comparision
    # ---------------------------------------------------------
    if args.visual_comparison:
        plot_segmentation_comparison(model, val_loader, device, )


    # ---------------------------------------------------------
    # Plot NDVI similarity plot for growth cycle
    # ---------------------------------------------------------
    if args.NDVI:
        print("🔎 Running Phenological Analysis...")
        plot_phenological_confusion(
            dataloader=val_loader,
        )
        print("✅ Analysis Complete. Check output folder.")

    # ---------------------------------------------------------
    # Plot Confusion matrix
    # ---------------------------------------------------------
    if args.confusion_matrix:
        cm = compute_and_plot_cm(
            model=model, 
            val_loader=val_loader, 
            device=device, 
            num_classes=num_classes, 
            class_names=class_names
        )

    # ---------------------------------------------------------
    # Per class mIoU Analysis
    # ---------------------------------------------------------
    if args.test_per_class:
        check_class_imbalance(model, val_loader, device=device, num_classes=num_classes)

    # ---------------------------------------------------------
    # Energy Efficiency Analysis
    # ---------------------------------------------------------
    if args.test_energy:
        measure_energy_efficiency_full(model, val_loader, device, disable_tqdm=args.no_progress_bar)

    # ---------------------------------------------------------
    # Accuracy Evaluation (mIoU)
    # ---------------------------------------------------------
    print("\n--- 🎯 Starting Accuracy Evaluation ---")
    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=ignore_index).to(device)
    oa_metric = MulticlassAccuracy(num_classes=num_classes, average='micro', ignore_index=ignore_index).to(device)

    #---------------------------------------------------------
    # Speed Test & Single Location Inference
    # ---------------------------------------------------------
    if args.inference:
        mIoU_rank = []
        
        # 1. Reset Model & Metrics
        reset_net(model)
        model.eval()

        single_miou = MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=ignore_index).to(device)
        
        with torch.no_grad():
            for index, batch in enumerate( tqdm(val_loader, desc="Evaluating Accuracy", disable=args.no_progress_bar) ):
                x = batch['sequence'].to(device)
                dates = batch['dates'].to(device)
                y = batch['labels'].to(device)

                # inference
                logits = model(x, dates)
                preds = torch.argmax(logits, dim=1)

                # Update Metric
                miou_metric.update(preds, y)
                oa_metric.update(preds, y)

                # calculate miou for this specific location
                single_miou.update(preds, y)
                current_score = single_miou.compute().item()
                single_miou.reset()
                mIoU_rank.append({"score": current_score, "index": index})

                # Reset SNN states (Voltage = 0)
                reset_net(model)


        # sort
        sort_score = sorted(mIoU_rank, key=lambda x: x["score"], reverse=True)
        top_10 = sort_score[:10]
        
        print(f"\n=========================================")
        print(f"🏆 Top 10 Best Performing Locations:")
        for item in top_10:
            # We use the index to look up the exact filename in your validation dataframe
            filename = val_df.iloc[item['index'], 0] 
            print(f"File: {filename} | mIoU: {item['score']:.4f}")
        print(f"=========================================\n")

        final_miou = miou_metric.compute().item()
        final_oa = oa_metric.compute().item()
        print(f"\n=========================================")
        print(f"🏆 Final Test mIoU: {final_miou:.4f}")
        print(f"🎯 Final Test OA: {final_oa:.4f}")
        print(f"=========================================\n")

    if args.deploy_inference:
        import time
        reset_net(model)
        model.eval()

        single_miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=ignore_index).to(device)
        single_oa_metric = MulticlassAccuracy(num_classes=num_classes, average='micro', ignore_index=ignore_index).to(device)

        try:
            batch = next(iter(val_loader))
        except StopIteration:
            print("Error: Validation is empty!")
            return
        
        x = batch['sequence'].to(device)
        dates = batch['dates'].to(device)
        y = batch['labels'].to(device)

        start_time = time.time()
        with torch.no_grad():
            logits = model(x, dates)
            preds = torch.argmax(logits, dim=1)

            single_miou_metric.update(preds, y)
            single_oa_metric.update(preds, y)
        
        end_time= time.time()
        duration = end_time - start_time

        # 5. Compute Scores
        final_miou = single_miou_metric.compute().item()
        final_oa = single_oa_metric.compute().item()

        print(f"✅ Inference Finished!")
        print(f"⏱️ Time taken for 1 Location: {duration:.4f} seconds")
        print(f"=========================================")
        print(f"🏆 Single Sample mIoU: {final_miou:.4f}")
        print(f"🎯 Single Sample OA:   {final_oa:.4f}")
        print(f"=========================================\n")

        

if __name__ == "__main__": # only run if execute python command.
    main() 
