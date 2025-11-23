# keep in mind that we only need the attention mechanism.
# so later on need to change those into all spike.

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
)
import torch.nn.functional as F
from timm.models.layers import DropPath

# BN for the RepConv because pad 0 will different after BN so to make test and train consistent we use this func
class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels, #the number of pixels to add to become the pad
        num_features, # the input features
        eps=1e-5, # epsilon, tiny number for the formula to not crash
        momentum=0.1, # how fast layer learns
        affine=True,  # can shift data when normalization
        track_running_stats=True # keep running average for inference.
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels
    
    # post padding, BN->padding
    def forward(self, input):
        output = self.bn(input)

        # calculating the values
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            # put the values 0 after BN into the pad
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.bias

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

# Reparameterization Conv, complex does training but simple for inferencing
class RepConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channels)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


# a down sampling class has in- out, detach, kernel, stride, pad
# this one has first_layer bool too.

class MS_Downsampling(nn.Module):
    def __init__(
        self,
        detach_reset=True,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1, # for every pixel get centered
        first_layer = True,
    ):
        super().__init__() # hierachy from parent class (nn.Module)

        # common downsampling is conv->bn->activation
        # bn to center data at a value(like 0) then learn to which value to center is the best
        # activation then put them in a range for non linearity.

        # conv
        self.encode_conv = nn.Conv2d(
            in_channels=in_channels, # same name then no need = if value dont need to change
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # bn
        self.bn = nn.BatchNorm2d(embed_dims)

        # lif only activate if not first layer
        if not first_layer:
            self.encode_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=detach_reset, backend='cupy'
            )

    def forward(self, x):
        T, B, _, H, W = x.shape

        # check if second layer
        if hasattr(self, 'encode_lif'):
            x = self.encode_lif(x)

        x = self.encode_conv(x.flatten(0, 1)) # conv process B first so merge T into B
        # bn -> reshape 5d -> contiguous
        x = self.bn(x).reshape(T, B, -1, H, W).contiguous()

        return x
    
"MS_Attention_RepConv - 2d attention"
class MS_Attention_RepConv(nn.Module):
    def __init__(
        self,
        dim, # transformer doesnt transform the input size
        num_heads=8,
        detach_reset=True,

        qkv_bias=False,
        qk_scale=None,
        sim_mode='dot',

        attn_drop=0.0,
        proj_drop=0.0,   # usually be applied in the final output of the attn block.
        sr_ratio=1.0
    ):
        super().__init__()
        assert(
            dim % num_heads == 0
        ), f"dim {dim} should be perfectly divided to num heads {num_heads}!"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads # for normalization
        self.scale = 0.125 # scaling down final output of attn
        self.sim_mode = sim_mode

        # the encoder neuron
        self.head_lif = MultiStepLIFNode( #with keyword, order doesn matter
            tau=2.0, detach_reset=detach_reset, backend='cupy'
        )

        # Q, K, V matricies
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        # now turn those QKV back to spike matrices
        self.q_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend="cupy"
        )

        self.k_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend='cupy'
        )

        self.v_lif = MultiStepLIFNode(
            tau=2.0, detach_reset=detach_reset, backend='cupy'
        )

        # a neuron to perform attention
        # the attn_lif will have 2 modes for dot and hamming
        if self.sim_mode =='dot':
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=detach_reset, backend='cupy', v_threshold=0.5
            )
        elif self.sim_mode == 'hamming':
            self.attn_bn = nn.BatchNorm2d(dim)
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=detach_reset, v_threshold=1.0, v_reset=0.0,
            )
        else: 
            raise NotImplementedError

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H*W # number of tokens for transformer

        # turn analog into spike
        x = self.head_lif(x) 
        # process Q, K, V
        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        
        # turn  QKV into spikes 
        q = self.q_lif(q).flatten(3) # flatten into 1 d
        q = ( # repairing for attn  
            q.transpose(-1, -2) # change to T, B, N, C => transformer format
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .transpose(2, 3) # transpose swap 2, permute reorder all.
            .contiguous()
        )  # Shape: (T, B, num_heads, N, head_dim)

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2) # T, B, N, C
            .reshape(T, B, N, self.num_heads, C// self.num_heads) # parrallel num head
            .permute(0, 1, 3, 2 ,4) # T, B, num_head, N, head_dim
            .contiguous()
        )   

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(2, 3)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .transpose(2, 3)
            .contiguous()
        )

        # calculate the attention map 
        # base on 2 modes
        if self.sim_mode == 'dot':
            x = k.transpose(-2, -1) @ v   # for pytorch to use matmul its only cares about last 2 dims
            x = (q @ x) * self.scale # x shape: (T, B, num_head, N, dim)
        elif self.sim_mode == 'hamming':
            x = (2 * k - 1).transpose(-2, -1) @ v
            x = (2 * q - 1) @ x
            x = x / (2 * self.head_dim)
        else: 
            raise NotImplementedError

        # now perform reshape and stacking
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()  # after permute or transpose should put contiguous for safety
        
        if self.sim_mode == 'dot':
            pass
        elif self.sim_mode == "hamming":
            x = x.view(T, B, C, H, W)
            x = self.attn_bn(x.flatten(0, 1)).reshape(T, B, C, H, W)
        else:
            raise NotImplementedError
        
        x = self.attn_lif(x).reshape(T, B, C, H, W) # now it stack but still need to process 
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x

"MLP block for the architecture"
class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        detach_reset=True,
        drop=0.0,
    ):
        super().__init__()

        # initilize hidden and out 
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # first MLP block
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1) # this layer will expand the attention map
        self.fc1_bn = nn.BatchNorm1d(hidden_features)  # put in the correct dim 
        self.fc1_lif = MultiStepLIFNode(  # turn into spike again
            detach_reset=detach_reset, tau=2.0, backend='cupy'
        )

        # the second block of MLP
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(
            detach_reset=detach_reset, tau=2.0, backend='cupy'
        )

        # save the variables 
        self.c_hidden= hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H*W
        
        # MLP is lif->conv->bn
        x = self.fc1_lif(x).flatten(3).flatten(0, 1) # shape: T*B, C, N
        x = self.fc1_conv(x) # MLP Conv works as 3d
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous() # put in contiguous for sure for the LIF

        x = self.fc2_lif(x).flatten(0, 1) # shape: T*B, C, N
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


"the MS_Block - the managing transformer block"
class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        detach_reset=True,
        mlp_ratio=4.0, # how much to expand in mlp
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,    #projection drop: the last drop layer
        attn_drop=0.1,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        sr_ratio = 1.0,
        attn_mode="2D_dot", # attn mode cause the model has 3 modes
    ):
        super().__init__()

        # now the attn mode
        if attn_mode == "2D_dot":
            print("2D attn mode is used.")
            self.attn = MS_Attention_RepConv(
                dim, 
                num_heads=num_heads,
                detach_reset=detach_reset,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                sim_mode='dot',
            )
        elif att_mode == '2D_ham':
            self.attn = MS_Attention_RepConv(
                dim, 
                num_heads=num_heads,
                detach_reset=detach_reset,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                sim_mode='hamming',
            )
        # currently not implementing those other attention
        else:
            raise NotImplementedError
        
        # initilize drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            detach_reset=detach_reset,
            drop=drop,
        )

    def forward(self, x):
        # applying residual connection, applying drop_path different from author
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


"temporal transformer"
class TemporalSpikingTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pe_dim=4,
        num_heads=8,
        temporal_depth=1,
        att_mode='2D_dot',
    ):
        super().__init__()
        # use PE library
        self.pe_date = nn.Embedding(num_embeddings=366, embedding_dim=pe_dim)
        # embedding layer
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels + pe_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        # attention_layer
        self.temporal_blocks = nn.ModuleList([
            MS_Block(
                dim=out_channels,
                num_heads=num_heads,
                detach_reset=True,
                att_mode=att_mode,
                # other use default parameters   
            )
            for _ in range(temporal_depth)
        ])
    
    def forward(self, x, dates):
        x = x.permute(1, 0, 2, 3, 4)
        dates = dates.permute(1, 0)
        T, B, C, H, W = x.shape

        # create dates PE
        pe = self.pe_date(dates)
        pe_dim = pe.shape[-1]
        
        # merge pe_date into x
        pe_img = pe.view(T, B, pe_dim, 1, 1).expand(T, B, pe_dim, H, W)
        x_pe = torch.cat([x, pe_img], dim=2)

        # pass through embedding layer 14 -> 64 channels
        x_flat = x_pe.flatten(0, 1)
        x_emb = self.embedding(x_flat)

        C_out = x_emb.shape[1]
        x = x_emb.view(T, B, C_out, H, W)

        # now change the shape so it process temporal feature
        x = x.permute(1, 3, 4, 2, 0).reshape(B*H*W, C_out, T)
        # unsqueeze to add a another dim=1 inside any position given
        x = x.unsqueeze(0).unsqueeze(-1) # (1, B*H*W, C_out, T, 1)

        # put in the attention block
        for block in self.temporal_blocks:
            x = block(x)
        
        # reshape back
        x = x.squeeze(0).squeeze(-1)
        x = x.view(B, H, W, C_out, T)
        x = x.permute(4, 0, 3, 1, 2).contiguous() #permute use indices
        
        return x

