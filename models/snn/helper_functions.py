import torch
import torch.nn as nn
from thop import profile
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import os

def get_norm_layer_2d(norm_type, channels):
    """
    Args:
        norm_type: 'bn' (Batch), 'gn' (Group/Layer)
        channels: number of features/channels
        dim_type: '1d' or '2d' (Tells us which BatchNorm to replace)
    """
    if norm_type == 'bn':
        return nn.BatchNorm2d(channels)
            
    elif norm_type == 'gn':
        # ✅ For Images: GroupNorm(1) is best
        return nn.GroupNorm(num_groups=1, num_channels=channels)
            
    else:
        raise NotImplementedError

def get_norm_layer_1d(norm_type, channels):
    """
    Args:
        norm_type: 'bn' (Batch), 'gn' (Group/Layer)
        channels: number of features/channels
        dim_type: '1d' or '2d' (Tells us which BatchNorm to replace)
    """
    if norm_type == 'bn':
        return nn.BatchNorm1d(channels)
            
    elif norm_type == 'gn':
        # ✅ For 1D Vectors: LayerNorm is safer (handles flat inputs)
        return nn.GroupNorm(num_groups=1, num_channels=channels)
            
    else:
        raise NotImplementedError


# --- Energy Calculation ---
class FiringRateMonitor:
    def __init__(self, model):
        self.hooks = []
        self.total_spikes = 0
        self.total_neurons = 0


        # attach hook into each LIF
        for name, layer in model.named_modules():
            # check is it LIF node to attach hook
            if hasattr(layer, 'tau') or hasattr(layer, 'v_threshold') or 'LIFNode' in layer.__class__.__name__:
                h = layer.register_forward_hook(self._hook_fn(name))
                self.hooks.append(h)
        
    def _hook_fn(self, name):
        def hook(module, input, output):
            # output shape: [T,B,C,H,W]
            spikes = output.sum().item() #due to binary values only 1 is been counted
            elements = output.numel() # now it counted every element like (T x B x...)

            self.total_spikes += spikes
            self.total_neurons += elements

        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove() #stop the hook logic
    def get_avg_firing_rate(self):
        if self.total_neurons == 0: return 0.0
        return self.total_spikes/ self.total_neurons


# Measure energy efficiency
def measure_energy_efficiency_full(model, dataloader, device, disable_tqdm=False):
    print("\n--- ⚡ Full Test Set Energy Analysis ---")

    # taking out the shape to calculate the capacity of the model
    dummy_batch = next(iter(dataloader))
    dummy_x = dummy_batch['sequence'].to(device)
    dummy_dates = dummy_batch['dates'].to(device)

    # Calculate Static FLOPs - maximum Possible Work
    model.eval() # thop look at architecture to count maximum capacity

    # 2. Calculate Static FLOPs (ANN Baseline)
    print("   ...Profiling Architecture FLOPs (this takes a second)...")
    batch_size = dummy_x.shape[0] # take batch shape
    flops_per_batch, params = profile(model, inputs=(dummy_x, dummy_dates), verbose=False)
    flops_per_sample = flops_per_batch / batch_size

    print(f"📦 Total Parameters: {params / 1e6:.2f} M")
    print(f"🧮 ANN Static FLOPs (per sample): {flops_per_sample / 1e9:.2f} G")

    # Measuring Firing Rate
    print(f"   ...Tracking Spikes across {len(dataloader)} batches...")
    
    monitor = FiringRateMonitor(model) # Attach spies

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Measuring Energy', leave=False, disable=disable_tqdm):
            x = batch['sequence'].to(device)
            dates = batch['dates'].to(device)
            # forward pass to trigger the hook and accum total spikes
            _ = model(x, dates)
        
    avg_firing_rate = monitor.get_avg_firing_rate()
    monitor.remove_hooks()

    print(f"🔥 Average Firing Rate (Full Set): {avg_firing_rate:.5f} ({avg_firing_rate*100:.3f}%)")

    # 5. Calculate Energy Ratio
    # ANN = 4.6 pJ (MAC)
    # SNN = 0.9 pJ (AC) * Sparsity
    
    energy_ann = flops_per_sample * 4.6
    energy_snn = flops_per_sample * avg_firing_rate * 0.9
    
    if energy_snn > 0:
        reduction = energy_ann / energy_snn
    else:
        reduction = 0 # Avoid divide by zero if model is dead
        
    print(f"🔋 ANN Energy / Sample: {energy_ann / 1e9:.2f} mJ")
    print(f"🔋 SNN Energy / Sample: {energy_snn / 1e9:.2f} mJ")
    print(f"🚀 Efficiency Gain: {reduction:.2f}x")
    
    return reduction


def check_class_imbalance(model, dataloader, device, num_classes=21):
    model.eval()
    
    # 1. Initialize Confusion Matrix (Confusion Matrix = num_classes x num_classes)
    # Rows = Ground Truth, Columns = Predictions
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    
    print("--- 📊 Analyzing Class Performance ---")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch['sequence'].to(device)
            dates = batch['dates'].to(device)
            y = batch['labels'].to(device) # Shape: [Batch, H, W]

            # Forward Pass
            logits = model(x, dates) 
            preds = torch.argmax(logits, dim=1) # Shape: [Batch, H, W]

            # 2. Flatten for easy counting
            y_flat = y.view(-1)
            preds_flat = preds.view(-1)

            # 3. Filter out ignore_index (usually -1 or 255) if necessary
            # mask = (y_flat >= 0) & (y_flat < num_classes)
            # y_flat = y_flat[mask]
            # preds_flat = preds_flat[mask]

            # 4. Update Confusion Matrix (Vectorized)
            # This maps (Target, Pred) pairs to a unique index
            indices = num_classes * y_flat + preds_flat
            counts = torch.bincount(indices, minlength=num_classes**2)
            
            # Reshape back to square matrix and add to total
            confusion_matrix += counts.view(num_classes, num_classes)
            
            # Optional: Stop after 50 batches to save time
            if i > 50: 
                break

    # 5. Calculate IoU per class
    # Intersection = Diagonal elements
    intersection = torch.diag(confusion_matrix)
    
    # Union = Sum of Rows + Sum of Cols - Intersection
    ground_truth_set = confusion_matrix.sum(dim=1)
    predicted_set = confusion_matrix.sum(dim=0)
    union = ground_truth_set + predicted_set - intersection

    # IoU = Intersection / Union
    iou_per_class = intersection / (union + 1e-6) # Add epsilon to avoid divide by zero

    # 6. Print Results
    print(f"\n{'Class ID':<10} | {'IoU':<10} | {'Status'}")
    print("-" * 40)
    
    for c in range(num_classes):
        iou = iou_per_class[c].item()
        
        status = ""
        if iou > 0.7: status = "🌟 Excellent"
        elif iou < 0.1: status = "⚠️ FAILED (Lazy Model)"
        elif iou == 0.0: status = "💀 DEAD"
        
        # Only print if the class actually exists in the GT
        if ground_truth_set[c] > 0:
            print(f"{c:<10} | {iou:.4f}     | {status}")
            
    # Calculate Mean IoU
    valid_classes = iou_per_class[ground_truth_set > 0]
    print("-" * 40)
    print(f"Mean IoU: {valid_classes.mean().item():.4f}")



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
            inputs = batch['sequence'].to(device)
            dates = batch['dates'].to(device)
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


# The NDVI test to show compare the phenology of similar parcels.

def plot_phenological_confusion(
    dataloader, 
    save_path="phenological_confusion.png",
    red_idx=2, 
    nir_idx=3, 
    num_samples=1000
):

    # Define the classes we want to compare
    class_map = {
        3: "Corn (Summer)",       # The confusing class
        18: "Sorghum (Summer)",   # The confusing class
        2: "Winter Wheat (Winter)" # Control class
    }

    # empty list and counter to store values 
    profiles = {k: [] for k in class_map.keys()}
    counts = {k: 0 for k in class_map.keys()}

    print(f"📊 Collecting spectral profiles for: {list(class_map.values())}...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scanning Dataset"):
            if isinstance(batch, dict):
                inputs = batch['sequence']
                targets = batch['labels']

            else:
                print("Unknown batch format. Skipping.")
                continue

            inputs = inputs.cpu()
            targets = targets.cpu()
            
            # Shape check and flattening
            if inputs.dim() == 5:
                B, T, C, H, W = inputs.shape

                # Permute to (Batch, H, W, Time, Channels) -> Flatten to (N, T, C)
                inputs = inputs.permute(0, 3, 4, 1, 2).reshape(-1, T, C)
                targets = targets.view(-1)

            # Extract samples for each target class
            for cls_id in class_map.keys():
                # If enough data, skip.
                if counts[cls_id] >= num_samples:
                    continue

                mask = (targets == cls_id)
                if mask.sum() == 0:
                    continue

                # Extract and store
                class_pixels = inputs[mask]
                n_take = min(num_samples - counts[cls_id], len(class_pixels))
                profiles[cls_id].append(class_pixels[:n_take].numpy())
                counts[cls_id] += n_take

            if all(c >= num_samples for c in counts.values()):
                break
    

    # --- PLOTTING ---
    print("📈 Generating NDVI Plot...")
    plt.figure(figsize=(10, 6))
    plt.title("Spectral Phenology Profile: The Source of Confusion", fontsize=14)
    plt.xlabel("Time Steps (Acquisition Dates)", fontsize=12)
    plt.ylabel("NDVI (Vegetation Health)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    colors = {3: 'green', 18: 'red', 2: 'blue'}
    styles = {3: '-', 18: '--', 2: ':'}
    
    # Plot each class
    for cls_id, name in class_map.items():
        if len(profiles[cls_id]) == 0:
            print(f"⚠️ Warning: No samples found for {name}")
            continue
            
        data = np.concatenate(profiles[cls_id], axis=0) # Shape: (N, T, C)
        
        # Calculate NDVI: (NIR - Red) / (NIR + Red)
        red_band = data[:, :, red_idx]
        nir_band = data[:, :, nir_idx]
        
        # Handle Potential Division by Zero
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)
        
        # Calculate Statistics
        mean_ndvi = np.mean(ndvi, axis=0)
        std_ndvi = np.std(ndvi, axis=0)
        x_axis = np.arange(len(mean_ndvi))
        
        # Plot
        plt.plot(x_axis, mean_ndvi, label=name, color=colors[cls_id], linestyle=styles[cls_id], linewidth=2)
        plt.fill_between(x_axis, mean_ndvi - 0.2*std_ndvi, mean_ndvi + 0.2*std_ndvi, color=colors[cls_id], alpha=0.1)

    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save directory check
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to: {save_path}")
