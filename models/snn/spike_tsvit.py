import torch
import torch.nn
from .snn_encoder import RecurrentSpikingEncoder
from .snn_transformer import MS_Block

class SpikeTSViT(nn.Module):
    def __init__(
        self,
        
    ):