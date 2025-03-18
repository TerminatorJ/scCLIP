import torch
import torch.nn as nn
from scCLIP.model.submodules.modules import AltAttention, GLUMlp
from scCLIP.model.submodules.moe import MoE
from scCLIP.model.submodules.layer_modules import DropPath, ScaleBiasLayer


class PjBlock(nn.Module):
    def __init__(self, 
                 dim=256, 
                 motif_dim=268,
                 num_heads=4, 
                 expand=4, 
                 attn_dropout=0.2, 
                 mlp_dropout=0.2, 
                 drop_path=0., 
                 activation='gelu', 
                 prenorm=True, 
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
        Projection of the gene sequence to the TF binding spaces.
        Args:
        moe: whether to use mixture of experts, if True, the block will have a gating network, otherwise, it will have a single feedforward network
        '''
        super().__init__(**kwargs)
        self.moe = moe
        self.moe_layer = MoE(dropout,activation,noisy_gating,num_experts,moe_input_size,moe_output_size,moe_hidden_size,moe_k)
        self.norm1 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.self_attn = AltAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)
        self.drop1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)#MaskedBatchNorm1d(dim, momentum=0.05, channels_last=True)
        self.mlp = GLUMlp(dim, expand, dropout=mlp_dropout, activation=activation)
        self.drop2 = DropPath(drop_path)

        self.prenorm = prenorm
        self.attn_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        
        #projection heads
        self.proj_head = nn.Linear(dim, motif_dim, bias=True)
        
        
    def forward(self, inputs, mask=None, alibi_bias=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        # import pdb; pdb.set_trace()
        x = self.self_attn(x,mask=mask,alibi_bias=alibi_bias)
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
                x = self.norm2(x)
                
            x = self.proj_head(x)
            return x, loss
        else:
            x = self.mlp(x)
            x = self.drop2(x)
            x = self.mlp_scale(x)
            x = x + attn_out
            if not self.prenorm:
                x = self.norm2(x)
            x = self.proj_head(x)
            return x
        
        
        
if __name__ == "__main__":
    x = torch.randn(2, 100, 256)
    mask = torch.ones(2, 100, dtype=torch.bool)
    model = PjBlock(dim=256, moe=False)
    y = model(x, mask)
    # import pdb; pdb.set_trace()
    
    #python -m scCLIP.model.submodules.projection
    