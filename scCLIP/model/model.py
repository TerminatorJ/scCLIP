import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from pathlib import Path
from scCLIP.model.submodules import Conv1DBlock,  AltBlock, MaskedConv2d, MoE, PjBlock
from scCLIP.model.settings import Settings


class SqueezeformerBlock(nn.Module):
    def __init__(self,
                 dim=256,
                 kernel_size=17,
                 groups=4,
                 num_heads=4,
                 conv_expand=4,
                 attn_expand=4,
                 num_conv_block=1,
                 num_attn_block=1,
                 num_moe_block=1,
                 conv_dropout=0.1,
                 attn_dropout=0.1,
                 mlp_dropout=0.1,
                 drop_path=0.1,
                 activation='swish',
                 prenorm=True,
                 cov=True,
                 use_flash_attn=False,
                 use_alibi=True,
                 moe=True, 
                 dropout=0.1,
                 noisy_gating=True, 
                 num_experts=5, 
                 moe_input_size=10,
                 moe_output_size=10,
                 moe_hidden_size=10,
                 moe_k=3,
                 **kwargs):

        super().__init__(**kwargs)
        self.num_moe_block = num_moe_block
        self.moe = moe
        self.cov = cov
        self.conv_blocks = nn.ModuleList([Conv1DBlock(dim,kernel_size,groups,1,1,conv_dropout,mlp_dropout,drop_path,conv_expand,activation,prenorm) for _ in range(num_conv_block)])
        self.attn_blocks = nn.ModuleList([AltBlock(dim,num_heads,attn_expand,attn_dropout,mlp_dropout,drop_path,activation,prenorm,moe,use_flash_attn,use_alibi,dropout,noisy_gating,num_experts,moe_input_size,moe_output_size,moe_hidden_size,moe_k) for _ in range(num_attn_block)])

    def forward(self, inputs, mask=None):
        x = inputs #(B,N,C)
        for block in self.attn_blocks:
            if self.moe:
                x, loss = block(x, mask=mask)
            else:
                x = block(x, mask=mask)
                loss = None
                
        if self.cov:
            for block in self.conv_blocks:
                x = block(x, mask=mask)
            return x, loss
        else:
            return x, loss

    def get_attn(self):
        for block in self.attn_blocks:
            # import pdb; pdb.set_trace()
            attns = block.self_attn.attn_score
        return attns

        
class SpaEncoder(nn.Module):
    def __init__(self,
                 dim=256,
                 motif_dim=286,
                 kernel_size=17,
                 groups=4,
                 num_heads=4,
                 num_layers=12,
                 conv_expand=4,
                 attn_expand=4,
                 num_conv_block=1,
                 num_attn_block=1,
                 num_moe_block=1,
                 conv_dropout=0.1,
                 attn_dropout=0.1,
                 mlp_dropout=0.1,
                 drop_path=0.1,
                 activation='swish',
                 prenorm=False,
                 cov=True,
                 use_flash_attn=True,
                 use_alibi=True,
                 moe=True, 
                 dropout=0.1,
                 noisy_gating=True, 
                 num_experts=5, 
                 moe_input_size=256,
                 moe_output_size=256,
                 moe_hidden_size=256,
                 moe_k=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.moe = moe
        self.prenorm = prenorm
        self.emb_dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        #The main structure of the model
        self.layers = nn.ModuleList(
                    [SqueezeformerBlock(dim,
                                        kernel_size,
                                        groups,
                                        num_heads,
                                        conv_expand,
                                        attn_expand,
                                        num_conv_block,
                                        num_attn_block,
                                        num_moe_block,
                                        conv_dropout,
                                        attn_dropout,
                                        mlp_dropout,
                                        drop_path,
                                        activation,
                                        prenorm,
                                        cov,
                                        use_flash_attn,
                                        use_alibi,
                                        moe, 
                                        dropout,
                                        noisy_gating, 
                                        num_experts, 
                                        moe_input_size,
                                        moe_output_size,
                                        moe_hidden_size,
                                        moe_k,) for _ in range(num_layers)])
        #The dense transformer layer for the supervised learning: gene -> motifs
        #This is kind of project head from one modality to the other one.
        self.projector_layer = PjBlock(dim,
                                       motif_dim, 
                                       num_heads, 
                                       attn_expand, 
                                       attn_dropout, 
                                       mlp_dropout,
                                       drop_path, 
                                       activation, 
                                       prenorm, 
                                       use_flash_attn,
                                       use_alibi,
                                       moe, 
                                       dropout, 
                                       noisy_gating, 
                                       num_experts, 
                                       moe_input_size, 
                                       moe_output_size, 
                                       moe_hidden_size, 
                                       moe_k)  
        
        #project the motif matrix to the gene space
        self.projector_back = nn.Linear(motif_dim, dim)
   
    def forward(self, seq_embed, mask=None):
        x = seq_embed
        x = self.emb_dropout(x)
        x = self.norm(x)
        # Apply the learned weight to the bpp matrix
        # bias = None
        #Supervised learning
        if self.moe:
            motif_output, loss = self.projector_layer(x, mask=mask)
        else:
            motif_output = self.projector_layer(x, mask=mask)
            loss = None
        #project the motif to normal space
        x = self.projector_back(motif_output)
        #input to the main structure
        x = x + seq_embed
        outputs = []
        attn_scores = []
        losses = []
        
        for layer in self.layers:
            x,loss = layer(x, mask=mask)
            attn_score = layer.get_attn()
            if mask is not None and hasattr(layer, 'compute_mask'):
                mask = layer.compute_mask(x, mask)

            outputs.append(x)
            attn_scores.append(attn_score)
            losses.append(loss)
        
        return motif_output, outputs, attn_scores, losses
    
    
if __name__ == '__main__':
    model = SpaEncoder(dim=256, cov=False, use_alibi=True, use_flash_attn=True)
    import pdb; pdb.set_trace()
    x = torch.randn(2, 100, 256).to(Settings.dtype)
    # summary(model, input_size = (100, 256))
    
    mask = torch.ones(2, 100, dtype=torch.bool)
    motif_matrix = torch.randn(2, 100, 286)#this is the motif matrix of the input
    motif_output, outputs, attn_scores, losses = model(x, mask=mask)
    import pdb; pdb.set_trace()
    print(model)
    #visualize the model
    make_dot(motif_output, params=dict(model.named_parameters())).render("model_architecture", format="png")
    import pdb; pdb.set_trace()

