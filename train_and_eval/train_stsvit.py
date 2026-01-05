import sys
import os
from torch.cuda.amp import GradScaler

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
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassJaccardIndex # mIoU score

from spike_data.pastis_dataset import PastisDataset
from spike_data.france_dataset import FranceDataset
from models.snn.spike_tsvit import SpikeTSViTMean, SpikeTSViTNoMean
from spikingjelly.clock_driven.functional import reset_net

from models.snn.loss_function import FocalLoss

# --- ARGUMENT PARSER ---
def get_args():
    parser = argparse.ArgumentParser(description="Train SpikeTSViT on PASTIS")

    # Paths
    parser.add_argument('--csv_path', type=str, 
                        default='/kaggle/working/DeepSatModels_SNN/configs/PASTIS24/splits/pastis_subset_10_percent.csv',
                        help="Path to the CSV file containing data paths")
    parser.add_argument('--val_csv_path', type=str, default=None, help="Optional: Path to Validation CSV. If None, splits csv_path.")
    parser.add_argument('--data_root', type=str, default="/kaggle/input", help="Root dir of dataset")
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/kaggle/working/spike_tsvit_checkpoint.pth',
                        help='Full path to save the best model checkpoint')
    parser.add_argument('--resume', type=str, default=None, help="Path to a checkpoint (.pth) to resume training from")
    parser.add_argument('--datasets', type=str, default='pastis', choices=['pastis', 'france'], help='choose the dataset')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--grad_accum_steps', type=int, default=1, help="Virtual batch size multiplier")

    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=64, help="Embedding dim")
    parser.add_argument('--heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--spatial_depth', type=int, default=1, help='Number of spatial blocks')
    parser.add_argument('--temporal_depth', type=int, default=1, help='Number of temporal blocks')
    parser.add_argument('--max_seq_len', type=int, default=10,
                        help="Fixed time length for input sequences - def=10")
    parser.add_argument('--model_type', type=str, default="mean", choices=['mean', 'no_mean'], help="The architecture type")

    parser.add_argument('--loss_type', type=str, default='standard',    
                        choices=['standard', 'weighted', 'focal'],
                        help="Choose loss function: 'standard' (CE), 'weighted' (CE + Weights), or 'focal' (Focus on hard examples)")
    parser.add_argument('--focal_a_weight', type=float, default=3.0, help="The alpha weight for focal loss")
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


# --- HELPER FUNCTIONS ---

def train_one_epoch(model, dataloader, optimizer, criterion, device, accum_steps, disable_tqdm=False, dynamic_ncols=True, scaler=None):
    model.train() # set model to train
    total_loss = 0.0
    optimizer.zero_grad() # set zero grad
    progress_bar = tqdm(dataloader, desc="Training", leave=False, disable=disable_tqdm, dynamic_ncols=True) # wrap dataloader act as iterator

    # run for each batch
    for i, batch in enumerate(progress_bar):
        # load from the dataset
        x = batch['sequence'].to(device)
        dates = batch['dates'].to(device)
        y = batch['labels'].to(device)

        # Amp utilized
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # clear->logit->computeLoss->backprobagation->learn->resetLIF
            logits = model(x, dates)
            loss = criterion(logits, y) # var to store loss history, cal grad to adjust weight

            # 2 Scale loss due to the fact crossentropy cals 4 imgs a time.
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        # 3. Conditional Update
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update() # Update the scale factor for next batch
            optimizer.zero_grad()

        reset_net(model) # delete the voltage left out of LIF

        total_loss += loss.item() * accum_steps
        if not disable_tqdm:
            progress_bar.set_postfix(loss=loss.item() * accum_steps)# text after the bar display instantly
        
    # Handle Leftovers
    if (len(dataloader) % accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        reset_net(model)

    return total_loss / len(dataloader)


def evaluate(model, dataloader, metric_fn, device, disable_tqdm=False, dynamic_ncols=True):
    model.eval()
    metric_fn.reset() # metric func-obj to know how good prediction ares

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False, disable=disable_tqdm, dynamic_ncols=True):
            x = batch['sequence'].to(device)
            dates = batch['dates'].to(device)
            y = batch['labels'].to(device)

            logits = model(x, dates)
            preds = torch.argmax(logits, dim=1) # argmax to has the correct class, dim 1 = C
            metric_fn.update(preds, y)
            reset_net(model)
        
    return metric_fn.compute().item()


# --- 3. MAIN EXECUTION ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- 🚀 Starting Training ---")
    print(f"📋 EXPERIMENT CONFIGURATION")
    print(f"{'='*40}")
    print(f"Datasets: {args.datasets}")

    if args.datasets == 'france':
        DatasetLoader = FranceDataset
        in_channels = 13
        num_classes = 21
        ignore_index = 20
        print(f"Dataset Selected: France dataset.")
    else: 
        DatasetLoader = PastisDataset
        in_channels = 10
        num_classes = 20
        ignore_index = 19
        print(f"Dataset Selected: PASTIS dataset.")
    
    # Paths
    print(f"📂 Paths:")
    print(f"   CSV Path:        {args.csv_path}")
    print(f"   Data Root:       {args.data_root}")
    print(f"   Checkpoint Path: {args.checkpoint_path}")
    if args.resume:
        print(f"   Resume From:     {args.resume}")

    # Hyperparameters
    print(f"\n⚙️  Hyperparameters:")
    print(f"   Batch Size:      {args.batch_size}")
    print(f"   Effective Batch Size: {args.batch_size * args.grad_accum_steps}")
    print(f"   Epochs:          {args.epochs}")
    print(f"   Learning Rate:   {args.lr}")

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

    # --- DATA SPLITTING LOGIC ---
    if args.val_csv_path:
        print(f"📂 Standard Mode: Using explicit Train/Val splits.")
        print(f"   Train: {args.csv_path}")
        print(f"   Val:   {args.val_csv_path}")

        train_df = pd.read_csv(args.csv_path, header=None)
        val_df = pd.read_csv(args.val_csv_path, header=None)
    else:       
        print(f"🧪 Experiment Mode: Randomly splitting single CSV.")
        print(f"   Source: {args.csv_path}")
        full_df = pd.read_csv(args.csv_path, header=None)
        train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        DatasetLoader(train_df, args.data_root, max_seq_len=args.max_seq_len, mode='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True
    )

    val_loader = DataLoader(
        DatasetLoader(val_df, args.data_root, max_seq_len=args.max_seq_len, mode='eval'),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True, pin_memory=True
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
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scaler = GradScaler()

    # focal loss or standard loss        
    if args.loss_type == 'focal':
        print("Mode: Focal loss ( Auto Focusing)")
        print(f"Focal alpha weight: {args.focal_a_weight}")

        if args.datasets == 'pastis':
            alpha_weights = [1.0] + [args.focal_a_weight] * 18 + [1.0]
        elif args.datasets == 'france':
            alpha_weights = [1.0] + [args.focal_a_weight] * 19 +[1.0]
            
        criterion = FocalLoss(alpha=alpha_weights, gamma=2.0, ignore_index=ignore_index)
    else:
        print("Mode: Standard Loss (Flat weights)")
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            

    criterion = criterion.to(device)

    metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=ignore_index).to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,   # cut it by half
        patience=8,   # wait for 3 epochs
    )

    # --- 3. RESUME LOGIC ---
    start_epoch = 0
    best_score = 0.0
    # Early stopping logic
    early_stopping_patience = 15
    early_stopping_counter = 0
    early_stopping_delta = 0.0001

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"🔄 Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device)

            # Load states
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # # check if checkpoint has then load not then restart
            # if 'scheduler_state_dict' in checkpoint:
            #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # else:
            #     print("⚠️ No scheduler state found. Scheduler will restart.")

            # load epoch and score
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint.get('best_score', 0.0)
            early_stopping_counter = 0
            print(f"✅ Loaded checkpoint (Epoch {start_epoch}, Best mIoU: {best_score:.4f})")
        else:
            print(f"⚠️ Checkpoint path '{args.resume}' not found! Starting from scratch.")
    

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, accum_steps=args.grad_accum_steps, disable_tqdm=args.no_progress_bar, scaler=scaler)
        val_miou = evaluate(model, val_loader, metric, device, disable_tqdm=args.no_progress_bar)
        
        # apply scheduler
        scheduler.step(val_miou)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val mIoU: {val_miou:.4f}")

        # Save Best (Weights Only is fine, or Full Dict)
        if val_miou > (best_score + early_stopping_delta):
            early_stopping_counter = 0
            best_score = val_miou
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"   🎉 New Best Model Saved! (mIoU: {best_score:.4f})")
        else:
            early_stopping_counter += 1
            print(f"   ⏳ No improvement. Patience: {early_stopping_counter}/{early_stopping_patience}")
            # kill switch
            if early_stopping_counter >= early_stopping_patience:
                print(f"   🛑 Early Stopping Triggered! Model hasn't improved for {early_stopping_patience} epochs.")
                break

        # Create a full checkpoint dictionary
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score,
            'early_stopping_counter': early_stopping_counter,
        }

        # Save Latest (Full Dict for Resuming)
        latest_path = args.checkpoint_path.replace('.pth', '_latest.pth')
        torch.save(checkpoint_dict, latest_path)

if __name__ == "__main__": # only run if execute python command.
    main() 
