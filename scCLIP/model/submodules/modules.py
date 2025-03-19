from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
from pathlib import Path
from scCLIP.model.submodules.layer_modules import DropPath, ScaleBiasLayer
from scCLIP.model.submodules.masked_batchnorm import MaskedBatchNorm1d
from scCLIP.model.submodules.masked_conv import MaskedConv1d
from scCLIP.model.submodules.moe import MoE
from scCLIP.model.submodules.alibi import get_alibi
from scCLIP.model.submodules.activations import get_act_fn
from scCLIP.model.settings import Settings
from scCLIP.model.submodules.flash_attention2 import MHA as FlashMHA

class GLU(nn.Module):
    def __init__(self, dim: int, activation: str = 'sigmoid') -> None:
        super(GLU, self).__init__()
        self.dim = dim
        self.activation = get_act_fn(activation)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * self.activation(gate)

class Mlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: str = 'gelu'
    ) -> None:
        super(Mlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.act = get_act_fn(activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)

        return x

class GLUMlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: str = 'gelu'
    ) -> None:
        super(GLUMlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.glu = GLU(dim=-1, activation=activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand // 2, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.glu(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)

        return x


class MaskedSoftmax(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # self.softmax = nn.Softmax(self.dim)

    def forward(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            # adder = (1.0 - mask.to(inputs.dtype)) * (
            #     torch.finfo(inputs.dtype).min
            # )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            # inputs += adder
            mask = mask.to(Settings.device)
            inputs = inputs.masked_fill(~mask, torch.finfo(inputs.dtype).min)
        return F.softmax(inputs, dim=self.dim)#, dtype=torch.float32)

    
class AltAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0, use_alibi=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        # self.alibi_bias = get_alibi(x.size(1), num_heads).to(dtype=x.dtype, device=x.device).repeat(x.size(0), 1, 1, 1)
        self.use_alibi = use_alibi
        self.attn_score = None
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        # self.proj_drop = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        # import pdb; pdb.set_trace()
        qkv = self.qkv(inputs)# B x L X D -> B x L X 3D
        qkv = qkv.view(-1, inputs.shape[1], self.num_heads, self.dim * 3 // self.num_heads).permute(0, 2, 1, 3) # B X H X L X 3D/H
        q, k, v = qkv.split([self.dim // self.num_heads] * 3, dim=-1)# B X H X L X D/H

        if mask is not None:
            mask = mask[:, None, None, :] # B X 1 X 1 X L
        # import pdb; pdb.set_trace()
        #calculating the attention score
        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
        
        #attention score add with bias
        if self.use_alibi:
            alibi_bias = get_alibi(inputs.size(1), self.num_heads).to(dtype=inputs.dtype, device=inputs.device).repeat(inputs.size(0), 1, 1, 1)
            attn = attn.type_as(alibi_bias)
            attn += alibi_bias
        # import pdb; pdb.set_trace()
        attn = MaskedSoftmax(dim=-1)(attn, mask=mask)#.to(q.dtype)
        self.attn_score = attn
        attn = self.attn_drop(attn)

        x = attn @ v # B X H X L X D/H
        x = x.permute(0, 2, 1, 3).reshape(-1, inputs.shape[1], self.dim)# B X L X D
        #feedforward# B X L X D
        #getting the final Z
        x = self.proj(x)# B X L X D
        # x = self.proj_drop(x)
        return x

class AltBlock(nn.Module):
    def __init__(self, 
                 dim=256, 
                 num_heads=4, 
                 expand=4, 
                 attn_dropout=0.2, 
                 mlp_dropout=0.2, 
                 drop_path=0., 
                 activation='gelu', 
                 prenorm=True, 
                 use_flash_attn=False,
                 use_alibi=False,
                 moe=True, 
                 dropout=0.1,
                 noisy_gating=True, 
                 num_experts=5, 
                 moe_input_size=10,
                 moe_output_size=10,
                 moe_hidden_size=10,
                 moe_k=3,
                 **kwargs):

        '''
        Args:
        moe: whether to use mixture of experts, if True, the block will have a gating network, otherwise, it will have a single feedforward network
        '''
        super().__init__(**kwargs)
        self.moe = moe
        self.use_flash_attn = use_flash_attn
        self.moe_layer = MoE(dropout,activation,noisy_gating,num_experts,moe_input_size,moe_output_size,moe_hidden_size,moe_k)
        self.norm1 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.dtype = self.norm1.weight.dtype
        #define different types of attention mechanism; whether to use gpu effective attention or normal attention
        if use_flash_attn:
            # import pdb; pdb.set_trace()
            self.self_attn = FlashMHA(embed_dim=dim,
                                      num_heads=num_heads,
                                      use_flash_attn=use_flash_attn,
                                      dropout=attn_dropout,
                                      use_alibi=use_alibi,
                                      device=Settings.device,
                                      dtype=Settings.dtype,
                                      )
           
        else:    
            self.self_attn = AltAttention(dim=dim,use_alibi=use_alibi,num_heads=num_heads,dropout=attn_dropout)
        self.drop1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.mlp = GLUMlp(dim, expand, dropout=mlp_dropout, activation=activation)
        self.drop2 = DropPath(drop_path)

        self.prenorm = prenorm
        self.attn_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        
    def forward(self, inputs, mask=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        # import pdb; pdb.set_trace()
        if self.use_flash_attn:
            x = self.self_attn(x)
        else:
            x = self.self_attn(x,mask=mask)
        x = self.drop1(x)
        x = self.attn_scale(x)
        x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        attn_out = x

        if self.prenorm:
            x = self.norm2(x)
        if self.moe:
            x, loss = self.moe_layer(x)
            x = self.drop2(x)
            x = self.mlp_scale(x)
            x = x + attn_out
            if not self.prenorm:
                x = x.to(Settings.dtype)
                x = self.norm2(x)
            return x, loss
        else:
            x = self.mlp(x)
            x = self.drop2(x)
            x = self.mlp_scale(x)
            x = x + attn_out
            if not self.prenorm:
                x = self.norm2(x)
            return x
        
class Conv1DBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=17,
                 groups=4,
                 dilation=1,
                 stride=1,
                 conv_dropout=0.0,
                 mlp_dropout=0.0,
                 drop_path=0.0,
                 expand=4,
                 activation='swish',
                 prenorm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.prenorm = prenorm
        self.stride = stride

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.glu = GLU(dim=-1, activation=activation)
        self.expand_conv = nn.Linear(dim, 2*dim)
        self.conv = MaskedConv1d(dim, dim, kernel_size=kernel_size, groups=groups)
        self.conv_norm = MaskedBatchNorm1d(dim, momentum=0.05)
        self.conv_act = get_act_fn(activation)
        self.conv_proj = nn.Linear(dim, dim)
        self.mlp = GLUMlp(dim, expand, mlp_dropout, activation=activation)
        self.conv_dropout = nn.Dropout(conv_dropout)
        # self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.drop1 = DropPath(drop_path)
        self.drop2 = DropPath(drop_path)
        self.conv_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)

    def compute_mask(self, inputs, mask=None):
      if mask is not None:
        if self.stride > 1:
          mask = mask[:,::self.stride]
      return mask

    def forward(self, inputs, mask=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.expand_conv(x)
        x = self.glu(x)
        x = x.permute(0,2,1)
        x = self.conv(x,mask=mask)
        mask = self.compute_mask(inputs,mask)
        x = self.conv_norm(x,mask=mask)
        x = self.conv_act(x)
        x = self.conv_dropout(x)
        x = x.permute(0,2,1)
        x = self.conv_proj(x)
        x = self.drop1(x)
        x = self.conv_scale(x)
        if self.stride == 1:
            x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)

        conv_out = x
        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        # x = self.mlp_dropout(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        if self.stride == 1:
            x = x + conv_out
        if not self.prenorm:
            x = self.norm2(x)
        return x
    
