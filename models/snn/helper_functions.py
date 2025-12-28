import torch
import torch.nn as nn
from thop import profile
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from spike_data.pastis_dataset import PASTIS_CLASSES

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


def check_class_imbalance(model, dataloader, device, num_classes=21, ignore_index=19):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    
    print(f"--- 📊 Analyzing Class Performance (ignoring index {ignore_index}) ---")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch['sequence'].to(device)
            dates = batch['dates'].to(device)
            y = batch['labels'].to(device)

            logits = model(x, dates) 
            preds = torch.argmax(logits, dim=1)

            y_flat = y.view(-1)
            preds_flat = preds.view(-1)

            # --- FIX 1: ACTUAL FILTERING ---
            # Create a mask that is TRUE only for valid classes (not 19)
            mask = (y_flat != ignore_index)
            
            # Apply mask to keep only valid pixels
            y_flat = y_flat[mask]
            preds_flat = preds_flat[mask]

            if len(y_flat) == 0: continue # Skip if batch was all void

            indices = num_classes * y_flat + preds_flat
            counts = torch.bincount(indices, minlength=num_classes**2)
            confusion_matrix += counts.view(num_classes, num_classes)
            
            # --- FIX 2: REMOVE THE BREAK ---
            # if i > 50: break  <-- Delete this to see Rare Class 18!

    # Calculate IoU
    intersection = torch.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(dim=1)
    predicted_set = confusion_matrix.sum(dim=0)
    union = ground_truth_set + predicted_set - intersection
    iou_per_class = intersection / (union + 1e-6)

    print(f"\n{'Class ID':<10} | {'IoU':<10} | {'Status'}")
    print("-" * 40)
    
    # Loop up to ignore_index (so we stop before printing 19)
    # OR range(num_classes) if you want to verify 19 is gone
    for c in range(num_classes):
        # Skip the void class explicitely in print if you want
        if c == ignore_index: continue

        iou = iou_per_class[c].item()
        
        status = ""
        if iou > 0.7: status = "🌟 Excellent"
        elif iou < 0.1: status = "⚠️ FAILED"
        
        if ground_truth_set[c] > 0:
            print(f"{c:<10} | {iou:.4f}     | {status}")
        else:
            # Helps verify if Class 18 is truly missing or just empty
            print(f"{c:<10} | {'---':<10} | ❌ No GT Samples Found")

    valid_mask = (ground_truth_set > 0) & (torch.arange(num_classes, device=device) != ignore_index)
    print("-" * 40)
    print(f"Mean IoU: {iou_per_class[valid_mask].mean().item():.4f}")



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

# ----------------
# Visualizing 3 pictures, Ground truth, prediction and Error.
# -----------------
PASTIS_PALETTE = {
    0:  (0.0, 0.0, 0.0),       # Background -> Black
    1:  (1.0, 0.84, 0.0),      # Corn -> Gold
    2:  (0.87, 0.72, 0.53),    # Wheat -> Wheat color
    3:  (0.2, 0.8, 0.2),       # Winter Barley -> Green
    4:  (0.55, 0.27, 0.07),    # Rapeseed -> SaddleBrown
    5:  (1.0, 0.0, 1.0),       # Sunflower -> Magenta
    6:  (0.5, 0.0, 0.5),       # Sugar Beet -> Purple
    7:  (0.0, 0.0, 1.0),       # Meadow -> Blue
    8:  (0.0, 0.5, 0.5),       # Forest -> Teal
    9:  (0.5, 0.5, 0.5),       # Potato -> Gray
    10: (0.6, 0.4, 0.2),       # Soya -> Brown
    11: (1.0, 0.5, 0.0),       # Fodder -> Orange
    12: (0.8, 0.8, 0.0),       # Triticale -> Olive
    13: (0.8, 0.0, 0.0),       # Durum Wheat -> Dark Red
    14: (0.0, 1.0, 0.0),       # Fruits/Veg -> Lime
    15: (0.4, 0.2, 0.6),       # Vegetables -> Violet
    16: (0.9, 0.6, 0.6),       # Legumes -> Pink
    17: (0.3, 0.3, 0.0),       # Soybeans -> Dark Olive
    18: (0.0, 0.0, 0.5),       # Sorghum -> Navy
    19: (1.0, 1.0, 1.0),       # Void -> White
}

# --- 1. Fix the Color Map Creator ---
def create_cmap(num_classes=20):
    # FIX: Cleaned up the list comprehension
    colors = [PASTIS_PALETTE.get(i, (0, 0, 0)) for i in range(num_classes)]
    return mcolors.ListedColormap(colors)

def plot_segmentation_comparison(model, loader, device, num_samples=3, save_dir="output"):
    """
    Plots: Ground Truth | Prediction | Error Map
    """
    os.makedirs(save_dir, exist_ok=True)

    
    model.eval()
    cmap = create_cmap(20)

    # Get a batch
    batch = next(iter(loader))
    x = batch['sequence'].to(device)
    dates = batch['dates'].to(device)
    y_true = batch['labels'].to(device)

    with torch.no_grad():
        logits = model(x, dates)
        y_pred = torch.argmax(logits, dim=1) # [B, H, W]

    # Convert to CPU numpy for plotting
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # FIX: Handle PASTIS_CLASSES being a list or dict
    if isinstance(PASTIS_CLASSES, dict):
        idx_to_name = PASTIS_CLASSES
    else:
        idx_to_name = {i: name for i, name in enumerate(PASTIS_CLASSES)}

    for idx in range(min(num_samples, x.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # --- 1. Ground Truth ---
        im1 = axes[0].imshow(y_true_np[idx], cmap=cmap, vmin=0, vmax=19, interpolation='nearest')
        axes[0].set_title(f'Ground Truth (Sample {idx})', fontsize=14)
        axes[0].axis('off')

        # --- 2. Prediction ---
        # FIX: You were plotting y_true_np again! Changed to y_pred_np.
        # FIX: Fixed the syntax error (dot -> comma)
        im2 = axes[1].imshow(y_pred_np[idx], cmap=cmap, vmin=0, vmax=19, interpolation='nearest')
        axes[1].set_title(f'S-TSViT Prediction', fontsize=14)
        axes[1].axis('off')

        # --- 3. Error Map (Difference) ---
        error_mask = (y_pred_np[idx] != y_true_np[idx]).astype(float)
        
        # Ignore Void class (19) errors
        void_mask = (y_true_np[idx] == 19)
        error_mask[void_mask] = 0

        # FIX: Variable name consistency (error_map vs error_cmap)
        error_cmap = mcolors.ListedColormap(['black', 'red'])

        axes[2].imshow(error_mask, cmap=error_cmap, vmin=0, vmax=1, interpolation='nearest')
        axes[2].set_title(f"Error Map (Red = Mismatch)", fontsize=14)
        axes[2].axis('off')

        # --- Legend ---
        # FIX: typo 'dix' -> 'idx'
        unique_classes = np.unique(np.concatenate((y_true_np[idx], y_pred_np[idx])))
        
        # FIX: Initialize the list!
        patches = []

        for c in unique_classes:
            if c == 19: continue
            color = PASTIS_PALETTE.get(c, (0, 0, 0))
            name = idx_to_name.get(c, f"Class {c}")
            patches.append(Patch(color=color, label=f'{c}: {name}'))
        
        # Only add legend if we have classes to show
        if patches:
            fig.legend(handles=patches, loc='center right', title="Crop Classes")
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

        # Save
        save_path = f"{save_dir}/comparison_sample_{idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
        plt.close()