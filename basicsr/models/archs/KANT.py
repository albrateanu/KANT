import numbers
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2

##########################################################################
## Utils

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

##########################################################################
## Attention
class MHSA(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.proj_in = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.proj_in = KolmogorovArnoldNetwork(input_channels=dim, hidden_size=dim)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # Change shape to (batch_size, H, W, C)
        qkv = self.proj_in(x)
        x = x.permute(0, 3, 1, 2).contiguous() 
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.proj_out(out)
        out = out.view(b, c, h, w)
        return out


##########################################################################
## Mixture of Experts
class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super(FFN, self).__init__()
        if hidden_channels is None:
            hidden_channels = in_channels * 2
        self.proj_in = nn.Conv2d(in_channels, hidden_channels*2, kernel_size=1)
        self.conv = nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=1)
        self.proj_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.proj_in(x)
        x = self.conv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.proj_out(x)
        return x

# RetinexFormer FFN  
class FFN2(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out
    
class MoE(nn.Module):
    def __init__(self, num_experts, in_channels, out_channels, hidden_channels=None):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        # self.experts = nn.ModuleList([FFN2(dim=in_channels) for _ in range(num_experts)])
        self.experts = nn.ModuleList([FFN(in_channels, out_channels, hidden_channels) for _ in range(num_experts)])
        self.gating_network = nn.Conv2d(in_channels, num_experts, kernel_size=1)

    def forward(self, x):
        gate_scores = self.gating_network(x)
        gate_scores = F.softmax(gate_scores, dim=1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch, num_experts, channels, h, w]
        output = torch.sum(gate_scores.unsqueeze(2) * expert_outputs, dim=1)

        return output

class ConvKAN(nn.Module):
    def __init__(self, in_channels, num_functions, out_channels, kernel_size=3):
        super(ConvKAN, self).__init__()
        self.num_functions = num_functions
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Each `psi` processes a single channel at a time but each outputs multiple channels
        # We set each to output `num_functions` channels
        self.psi = nn.ModuleList([
            nn.Conv2d(1, num_functions, kernel_size, padding=kernel_size//2, bias=False)
            for _ in range(in_channels)
        ])

        # Phi function: Convolution that aggregates all psi outputs to the desired output channels
        self.phi = nn.Conv2d(num_functions * in_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        batch_size, _, H, W = x.shape
        # Apply each psi to the corresponding channel and store results in a temporary tensor
        psi_outputs = []
        for idx, psi in enumerate(self.psi):
            # Apply each psi layer to the corresponding input channel
            psi_output = psi(x[:, idx:idx+1, :, :])  # single channel input
            psi_outputs.append(psi_output)
        
        # Concatenate all psi outputs along the channel dimension
        psi_outputs = torch.cat(psi_outputs, dim=1)

        # Apply phi function to aggregate all psi outputs
        output = self.phi(psi_outputs)
        return output

class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_channels, hidden_size=256):
        super(KolmogorovArnoldNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        
        # Create separate MLPs for each channel
        self.fc1 = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(input_channels)])
        self.fc2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(input_channels)])
        self.fc3 = nn.ModuleList([nn.Linear(hidden_size, 3) for _ in range(input_channels)])
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size, H, W, C = x.shape
        x = x.view(-1, C)  # Flatten the spatial dimensions
        
        outputs = []
        
        for i in range(self.input_channels):
            xi = x[:, i:i+1]
            xi = self.relu(self.fc1[i](xi))
            xi = self.relu(self.fc2[i](xi))
            xi = self.fc3[i](xi)
            outputs.append(xi)
        
        x = torch.cat(outputs, dim=1)
        x = x.view(batch_size, H, W, C*3)
        
        return x

class KANAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(KANAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj_in = KolmogorovArnoldNetwork(dim, dim)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def apply_kan(self, kan, x):
        batch_size, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # Change shape to (batch_size, H, W, C)
        x = kan(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # Change back to (batch_size, C, H, W)
        return x

    def forward(self, x):
        b,c,h,w = x.shape

        # qkv = self.proj_in(x)
        qkv = self.apply_kan(self.proj_in, x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.proj_out(out)
        out = out.view(b, c, h, w)
        return out

##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, num_experts, dim_feedforward=None, dropout=0.1, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = in_channels * 2

        # self.attention = RestorationMHSA(dim=in_channels, num_heads=num_heads)
        # self.attention = MHSA(dim=in_channels, num_heads=num_heads)#
        self.attention = KANAttention(dim=in_channels, num_heads=num_heads)
        self.norm1 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)
        self.moe = FFN2(dim=in_channels)
        # self.moe = MoE(num_experts, in_channels, in_channels, dim_feedforward)
        # self.moe = ConvKAN(in_channels=in_channels, out_channels=in_channels, num_functions=2)
        self.norm2 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)

    def apply_kan(self, kan, x):
        batch_size, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # Change shape to (batch_size, H, W, C)
        x = kan(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # Change back to (batch_size, C, H, W)
        return x
    
    def forward(self, x):
        # x = x + self.norm1(self.apply_kan(self.attention, x))
        x = x + self.norm1(self.attention(x))
        x = x + self.norm2(self.moe(x))
        return x


##########################################################################
## Model 26.36 GTMean - KAN attention
class KANT(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, num_blocks=None, stage=None):
        super(KANT, self).__init__()
        
        num_heads = 2
        num_experts = None # 5 tasks
        
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=1, padding='same')

        # First level
        self.transformer_block1_1 = TransformerBlock(n_feat, num_heads, num_experts)
        self.downsample1 = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1)
        # num_heads *=2
        # Second level
        self.transformer_block2_1 = TransformerBlock(n_feat * 2, num_heads, num_experts)
        self.transformer_block2_2 = TransformerBlock(n_feat * 2, num_heads, num_experts)
        self.downsample2 = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)
        num_heads *=2
        # Bottleneck level
        self.bottleneck_1 = TransformerBlock(n_feat * 4, num_heads, num_experts)
        self.bottleneck_2 = TransformerBlock(n_feat * 4, num_heads, num_experts)
        num_heads //=2
        # Second level (upsampling)
        self.upsample2 = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust2 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1)
        self.transformer_block_up2_1 = TransformerBlock(n_feat * 2, num_heads, num_experts)
        self.transformer_block_up2_2 = TransformerBlock(n_feat * 2, num_heads, num_experts)
        num_heads //=2
        # First level (upsampling)
        self.upsample1 = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)
        self.transformer_block_up1_1 = TransformerBlock(n_feat, num_heads, num_experts)

        # Final output convolution
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.conv_in(x)
        x_in = x
        # Downward path
        x1 = self.transformer_block1_1(x)
        x1_down = self.downsample1(x1)

        x2 = self.transformer_block2_1(x1_down)
        x2 = self.transformer_block2_2(x2)
        x2_down = self.downsample2(x2)

        # Bottleneck
        bn = self.bottleneck_1(x2_down)
        bn = self.bottleneck_2(bn)

        # Upward path
        x2_up = self.upsample2(bn)
        x2_up = torch.cat([x2_up, x2], dim=1)
        x2_up = self.channel_adjust2(x2_up)
        x2_up = self.transformer_block_up2_1(x2_up)
        x2_up = self.transformer_block_up2_2(x2_up)

        x1_up = self.upsample1(x2_up)
        x1_up = torch.cat([x1_up, x1], dim=1)
        x1_up = self.channel_adjust1(x1_up)
        x1_up = self.transformer_block_up1_1(x1_up)

        # Final output
        x_out = self.conv_out(x1_up)
        # x_out = self.conv_out(torch.cat([x1_up, x_in], dim=1))
        return x_out