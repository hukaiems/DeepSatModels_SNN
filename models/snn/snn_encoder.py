"the encoder for the file"
import torch
import torch.nn as nn
import math
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

class RecurrentSpikingEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, pe_dim=4):
        super().__init__()
        # 1. Positional Encoding 
        # turn date into vector (4 dims)
        self.pos_encoder = nn.Embedding(num_embeddings=366, embedding_dim=pe_dim)
        # 2. Spatial Processor
        #  A standard 2d Conv to detect those spatial like edges
        # it also stacks date vector with spatial data.
        self.conv = nn.Conv2d(in_channels + pe_dim, out_channels, kernel_size=3, padding=1)
        # 3. BN
        self.bn = nn.BatchNorm2d(out_channels)
        # 4. LIF neuron
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")

    def forward(self, x_sequence, dates):
        x_sequence = x_sequence.permute(1, 0, 2, 3 ,4)
        dates = dates.permute(1, 0) # Pytorch expect first dim is time.
        # get PE 
        T, B, C, H, W = x_sequence.shape
        dates_flat = dates.flatten()
        pos_encoding_flat = self.pos_encoder(dates_flat)
        pe_dim = pos_encoding_flat.shape[-1]
        # creating pe_img
        pos_encoding_img = pos_encoding_flat.view(T, B, pe_dim, 1, 1).expand(T, B, pe_dim, H, W)
        # now merge pe with channels dim
        x_with_pos = torch.cat([x_sequence, pos_encoding_img], dim=2)
        # flatten T* B for the conv2d
        x_flat = x_with_pos.flatten(0, 1)
        features_flat = self.bn(self.conv(x_flat)) # now turn into feature
        # unflatten the features to feed into LIF neuron
        features = features_flat.view(T, B, -1, H, W)
        # now feed into neuron
        spike_train = self.lif(features)
        self.lif.reset() # so time step in each batch is seperated.

        return spike_train   # Shape: T, B, C, H, W
        


        
