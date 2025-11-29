import torch.nn as nn

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
        return nn.LayerNorm(channels)
            
    else:
        raise NotImplementedError