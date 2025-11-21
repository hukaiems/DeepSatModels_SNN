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
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassJaccardIndex # mIoU score

from data.pastis_dataset import PastisDataset
from models.snn.spike_tsvit import SpikeTSViT
from spikingjelly.clock_driven.functional import reset_net

# --- ARGUMENT PARSER ---
def get_args():
    parser = argparse.ArgumentParser(description="Train SpikeTSViT on PASTIS")

    # Paths
    parser.add_argument('--csv_path', type=str, 
                        default='/kaggle/working/DeepSatModels_SNN/configs/PASTIS24/splits/pastis_subset_10_percent.csv',
                        help="Path to the CSV file containing data paths")
    parser.add_argument('--data_root', type=str, default="/kaggle/input", help="Root dir of dataset")
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/kaggle/working/spike_tsvit_checkpoint.pth',
                        help='Full path to save the best model checkpoint')
    # Hyperparameters
    parser.add_argument('batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=64, help="Embedding dim")
    parser.add_argument('--heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--spatial_depth', type=int, default=1, help='Number of spatial blocks')

    parser.add_argument('--max_seq_len', type=int, default=10,
                        help="Fixed time length for input sequences - def=10")

    return parser.parse_args()


# --- HELPER FUNCTIONS ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train() # set model to train
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False) # wrap dataloader act as iterator

    # run for each batch
    for batch in progress_bar:
        # load from the dataset
        x = batch['sequence'].to(device)
        dates = batch['dates'].to(device)
        y = batch['labels'].to(device)

        # clearGrad->logit->computeLoss->backprobagation->learn->resetLIF
        optimizer.zero_grad() # clear gradients from previous batch
        logits = model(x, dates)
        loss = criterion(logits, y) # var to store loss history, cal grad to adjust weight
        loss.backward()
        optimizer.step()
        reset_net(model) # delete the voltage left out of LIF

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item()) # text after the bar display instantly

    return total_loss / len(dataloader)


def evaluate(model, dataloader, metric_fn, device):
    model.eval()
    metric_fn.reset() # metric func-obj to know how good prediction ares

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
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
    print(f"    CSV: {args.csv_path}")
    print(f"    Save Path: {args.checkpoint_path}")
    print(f"    Batch Size: {args.batch_size}")

    # 1. Data setup
    full_df = pd.read_csv(args.csv_path, header=None)
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        PastisDataset(train_df, args.data_root, max_seq_len=args.max_seq_len),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
        # how many imgs process in 1 batch
        # shuffle indices of samples before creating the batch
        # workers: number of cpu processes run to load data
    )

    val_loader = DataLoader(
        PastisDataset(val_df, args.data_root, max_seq_len=args.max_seq_len),
        batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    # 2. Model Setup
    model = SpikeTSViT(
        in_channels=10,
        embed_dim=args.embed_dim,
        num_classes=20,
        num_heads=args.heads,
        spatial_depth=args.spatial_depth,
        pe_dim=4
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss() 
    metric = MulticlassJaccardIndex(num_classes=20, average='macro').to(device)

    # 3. Training loop
    best_score = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_miou = evaluate(model, val_loader, metric, device)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val mIoU: {val_miou:.4f}")

        # Create a full checkpoint dictionary
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score,
        }

        # Save Best (Weights Only is fine, or Full Dict)
        if val_miou > best_score:
            best_score = val_miou
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"   🎉 New Best Model Saved! (mIoU: {best_score:.4f})")

        # Save Latest (Full Dict for Resuming)
        latest_path = args.checkpoint_path.replace('.pth', '_latest.pth')
        torch.save(checkpoint_dict, latest_path)

if __name__ == "__main__": # only run if execute python command.
    main() 
