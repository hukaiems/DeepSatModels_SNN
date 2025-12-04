import torch
import torch.nn as nn
from thop import profile
from tqdm import tqdm

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