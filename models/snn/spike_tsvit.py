import torch
import torch.nn as nn
from .snn_transformer import MS_Block, TemporalSpikingTransformer

"the normal testing "
class SpikeTSViTMean(nn.Module):
    def __init__(
        self,
        in_channels=10,
        embed_dim=64,
        temporal_depth=1,
        spatial_depth=1,
        num_classes=20,
        att_mode="2D_dot",
    ):
        super().__init__()

        self.temporal_encoder = TemporalSpikingTransformer(
            in_channels=in_channels,
            out_channels=embed_dim,
            temporal_depth=temporal_depth,
            att_mode=att_mode,
        )

        self.spatial_encoder = nn.ModuleList([
            MS_Block( # transformer block keep the shape in and out the same.
                dim=embed_dim,
                att_mode=att_mode
            )
            for _ in range(spatial_depth)
        ])

        # Decoder head
        # A simple conv to project features into class scores
        # Havent down sample H-W so no need for a complex one
        self.decoder = nn.Sequential(  # increase the computation or power to process better
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(), # lets me think later on
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        ) 

    def forward(self, x, dates):
        B, T, C, H, W = x.shape
        
        # 1 temporal encoder
        x = self.temporal_encoder(x, dates) #shape now T, B, C, H, W

        # collapse time
        x = x.mean(dim=0)
        x = x.unsqueeze(0) # 1, B, C_out, H, W
        # 2 spatial encoder
        for block in self.spatial_encoder:
            x = block(x)
        
        x = x.squeeze(0) # B, C_Out, H, W
        logits = self.decoder(x)

        return logits

class SpikeTSViTNoMean(nn.Module):
    def __init__(
        self,
        in_channels=10,
        embed_dim=64,
        temporal_depth=1,
        spatial_depth=1,
        num_classes=20,
        att_mode="2D_dot",
    ):
        super().__init__()

        self.temporal_encoder = TemporalSpikingTransformer(
            in_channels=in_channels,
            out_channels=embed_dim,
            temporal_depth=temporal_depth,
            att_mode=att_mode,
        )

        self.spatial_encoder = nn.ModuleList([
            MS_Block( # transformer block keep the shape in and out the same.
                dim=embed_dim,
                att_mode=att_mode
            )
            for _ in range(spatial_depth)
        ])

        # Decoder head
        # A simple conv to project features into class scores
        # Havent down sample H-W so no need for a complex one
        self.decoder = nn.Sequential(  # increase the computation or power to process better
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(), # lets me think later on
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        ) 

    def forward(self, x, dates):
        B, T, C, H, W = x.shape
        
        # 1 temporal encoder
        x = self.temporal_encoder(x, dates) #shape now T, B, C, H, W

        # 2 spatial encoder
        for block in self.spatial_encoder:
            x = block(x)   # Shape: T, B, C, H, W
        
        # now collapsing T dim to perform the prediction
        x = x.mean(0) # B, C_Out, H, W
        logits = self.decoder(x)

        return logits
