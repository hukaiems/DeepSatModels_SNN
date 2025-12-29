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
import random

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
    save_path="output/triticale_vs_wheat_phenology.png", # Updated filename
    band_idx=3, 
    band_name="NIR Intensity (Normalized)",
    num_samples=1000,
    window_size=5
):
    # --- 1. CONFIGURATION ---
    # CORRECTION: Triticale is ID 10, Wheat is ID 2
    class_map = {
        10: "Winter Triticale (Class 10)",      # The Confusing Class
        17: "Mixed Cereal (Class 17)"        # The Control Class
    }

    # Setup storage
    profiles = {k: [] for k in class_map.keys()}
    counts = {k: 0 for k in class_map.keys()}

    print(f"📊 Scanning dataset for: {list(class_map.values())}...")

    # --- 2. DATA COLLECTION ---
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting Profiles"):
            inputs = batch['sequence'].cpu()
            targets = batch['labels'].cpu()

            # Flatten batch dimensions: [B, T, C, H, W] -> [N, T, C]
            if inputs.dim() == 5:
                B, T, C, H, W = inputs.shape
                # Permute to put H,W alongside Batch, then flatten
                inputs = inputs.permute(0, 3, 4, 1, 2).reshape(-1, T, C)
                targets = targets.view(-1)
            
            # Loop through our target classes
            for cls_id in class_map.keys():
                # Skip if we already have enough data for this class
                if counts[cls_id] >= num_samples: continue

                # Find pixels belonging to this class
                mask = (targets == cls_id)
                if mask.sum() > 0:
                    class_data = inputs[mask]
                    
                    needed = num_samples - counts[cls_id]
                    to_take = class_data[:needed]
                    
                    profiles[cls_id].append(to_take.numpy())
                    counts[cls_id] += len(to_take)

            # Break early if we have full sets
            if all(c >= num_samples for c in counts.values()):
                break

    # --- 2. SMOOTHING FUNCTION ---
    def moving_average(data, window_size):
        pad = window_size // 2
        padded = np.pad(data, (pad, pad), mode='edge')
        return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

    # --- 3. PLOTTING ---
    plt.figure(figsize=(12, 7))
    
    # CORRECTION: Updated keys to match class_map (10 and 2)
    # Triticale (Orange-ish to stand out), Wheat (Blue standard)
    colors = { 10: '#ff7f0e', 17: '#58E074'} 
    styles = { 10: '--', 17: '-'}
    
    found_any = False
    
    for cls_id, name in class_map.items():
        if len(profiles[cls_id]) == 0: 
            print(f"⚠️ Warning: No samples found for {name}")
            continue
        
        found_any = True
        
        # Concatenate all pixels
        data_block = np.concatenate(profiles[cls_id], axis=0)
        
        # Extract the specific band
        band_data = data_block[:, :, band_idx]
        
        # Calculate Mean and Std
        mean_profile = np.mean(band_data, axis=0)
        std_profile = np.std(band_data, axis=0)

        smooth_mean = moving_average(mean_profile, window_size)
        smooth_std = moving_average(std_profile, window_size)

        x_axis = np.arange(len(smooth_mean))

        # Plot Line
        plt.plot(x_axis, smooth_mean, label=name, 
                 color=colors[cls_id], linestyle=styles[cls_id], linewidth=3)
        
        # Plot Shadow
        plt.fill_between(x_axis, smooth_mean - 0.5*smooth_std, 
                         smooth_mean + 0.5*smooth_std, 
                         color=colors[cls_id], alpha=0.15)

    if found_any:
        plt.title(f"Spectral Profile Overlap: Triticale vs. Mixed Cereal\n({band_name})", fontsize=16)
        plt.xlabel("Time Steps (Season)", fontsize=14)
        plt.ylabel("Pixel Intensity (Normalized)", fontsize=14)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved Analysis Plot to {save_path}")
    else:
        print("❌ Failed to generate plot: No data found for selected classes.")

# USAGE:
# plot_phenological_confusion(val_loader, band_idx=3)

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

def normalize_for_display(img_tensor):
    """
    Revert back value from [0, 10k] to [0, 1] then multiply for 255 for rgb range.
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy() #[H, W, 3]
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip((img - p2) / (p98 - p2), 0, 1)
    return img

def create_cmap(num_classes=20):
    # FIX: Cleaned up the list comprehension
    colors = [PASTIS_PALETTE.get(i, (0, 0, 0)) for i in range(num_classes)]
    return mcolors.ListedColormap(colors)

def plot_segmentation_comparison(model, loader, device, num_samples=5, save_dir="output"):
    """
    Plots: Ground Truth | Prediction | Error Map
    """
    os.makedirs(save_dir, exist_ok=True)


    model.eval()
    cmap = create_cmap(20)

    # Get a random batch
    random_idx = random.randint(0, len(loader) - 1)
    data_iter = iter(loader)
    for _ in range(random_idx):
        next(data_iter)
    
    batch = next(data_iter)

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
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))


        # --- 0. Optical RGB 
        rgb_bands = x[idx, :, [2, 1, 0], :, :]
        median_rgb = torch.median(rgb_bands, dim=0)[0]
        rgb_display = normalize_for_display(median_rgb)

        axes[0].imshow(rgb_display)
        axes[0].set_title(f'Optical Image (Median)', fontsize=14)
        axes[0].axis('off')


        # --- 1. Ground Truth ---
        im1 = axes[1].imshow(y_true_np[idx], cmap=cmap, vmin=0, vmax=19, interpolation='nearest')
        axes[1].set_title(f'Ground Truth (Sample {idx})', fontsize=14)
        axes[1].axis('off')

        # --- 2. Prediction ---
        # FIX: You were plotting y_true_np again! Changed to y_pred_np.
        # FIX: Fixed the syntax error (dot -> comma)
        im2 = axes[2].imshow(y_pred_np[idx], cmap=cmap, vmin=0, vmax=19, interpolation='nearest')
        axes[2].set_title(f'S-TSViT Prediction', fontsize=14)
        axes[2].axis('off')

        # --- 3. Error Map (Difference) ---
        error_mask = (y_pred_np[idx] != y_true_np[idx]).astype(float)
        
        # Ignore Void class (19) errors
        void_mask = (y_true_np[idx] == 19)
        error_mask[void_mask] = 0

        # FIX: Variable name consistency (error_map vs error_cmap)
        error_cmap = mcolors.ListedColormap(['black', 'red'])

        axes[3].imshow(error_mask, cmap=error_cmap, vmin=0, vmax=1, interpolation='nearest')
        axes[3].set_title(f"Error Map (Red = Mismatch)", fontsize=14)
        axes[3].axis('off')

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
        plt.subplots_adjust(right=0.88)

        # Save
        save_path = f"{save_dir}/comparison_sample_{idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
        plt.close()



# --------------------------------------
# Analyzing temporal importance
#  -------------------------------------

def analyze_temporal_importance(model, loader, device, save_dir="output", target_class=9):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    print("Calculating Baseline Accuracy")
    correct = 0
    total = 0

    # for convienience we only use 50 batches
    subset_limit = 50
    val_data = []
    for i, batch in enumerate(loader):
        if i >= subset_limit: break
        val_data.append(batch)
    

    def evaluate_batch_for_class(data_list, mask_t=None):
        c, t = 0, 0
        with torch.no_grad():
            for batch in data_list:
                x = batch['sequence'].clone().to(device) # clone x so we wont corrupt data
                dates = batch['dates'].to(device)
                y_true = batch['labels'].to(device)

                # apply Occlusion (Masking)
                if mask_t is not None:
                    x[:, mask_t, :, :, :] = 0
                
                logits = model(x, dates)
                y_pred = torch.argmax(logits, dim=1)

                # score calculation
                mask = (y_true == target_class)

                if mask.sum() > 0:
                    c += (y_pred[mask] == y_true[mask]).sum().item()
                    t += mask.sum().item()
        return c / t if t >0 else 0

    baseline_acc = evaluate_batch_for_class(val_data, mask_t=None)
    print(f" Baseline Accuracy (Subset): {baseline_acc:.4f}")

    # Loop through time steps
    num_timesteps = val_data[0]['sequence'].shape[1]
    importance_scores = []

    print(f' Testing Importance of {num_timesteps} Time Steps...')

    for t in tqdm(range(num_timesteps)):
        masked_acc = evaluate_batch_for_class(val_data, mask_t=t)

        drop = baseline_acc - masked_acc
        importance_scores.append(drop)

    plt.figure(figsize=(12, 6))
    x_axis = np.arange(num_timesteps)
    
    # Plot bars
    # Use color to highlight positive (important) vs negative (noise)
    colors = ['red' if x > 0 else 'gray' for x in importance_scores]
    plt.bar(x_axis, importance_scores, color=colors, alpha=0.7)
    
    # Add a smooth trend line to see the "Season"
    # Simple moving average
    if len(importance_scores) > 5:
        smooth = np.convolve(importance_scores, np.ones(3)/3, mode='same')
        plt.plot(x_axis, smooth, color='black', linestyle='--', linewidth=2, label='Trend')

    plt.title("Temporal Feature Importance (Occlusion Sensitivity)", fontsize=16)
    plt.ylabel("Drop in Accuracy (Importance)", fontsize=14)
    plt.xlabel("Time Steps", fontsize=14)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    save_path = f"{save_dir}/temporal_importance_class_{target_class}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved Importance Plot to {save_path}")