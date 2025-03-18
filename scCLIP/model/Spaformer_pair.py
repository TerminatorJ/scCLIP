import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List
from torch import optim
import numpy as np
import math
import os
import sys
from pathlib import Path
from .model import *
import pickle


#prefix for the model
MASK_TOKEN = 2
CLS_TOKEN = 1
PAD_TOKEN = 0


class Spaformer(pl.LightningModule):
    def __init__(self, 
                 dim_model: int, 
                 nheads: int, 
                 nlayers: int, 
                 dropout: float,
                 masking_p: float, 
                 n_tokens: int,
                 n_atokens: int,
                 context_length: int,
                 lr: float, 
                 warmup: int, 
                 max_epochs: int,
                 pool: str = None,
                 mask_way: str = None,
                 outer_config: dict = None,
                 ):
        """
        Args:
            dim_model (int): Dimensionality of the model
            nheads (int): Number of attention heads
            nlayers (int): Number of layers
            dropout (float): Dropout rate
            masking_p (float): p value of Bernoulli for masking
            n_tokens (int): total number of tokens (WITHOUT auxiliar tokens), only the gene indices
            n_atokens (int): total number of auxiliar tokens
            context_length (int): length of the context, which means the fixed number of the input sequence
            lr (float): learning rate
            warmup (int): number of steps that the warmup takes
            max_epochs (int): number of steps until the learning rate reaches 0
            pool (str): could be None, 'cls' or 'mean'. CLS adds a token at the beginning, mean just averages all tokens. If not supervised task during training, is ignored
            outer_config (dict): the overall config of the model for training
        """
        super().__init__()
        # self.batch_length = None
        self.encoder = SpaEncoder(dim=dim_model , num_layers=nlayers, groups=dim_model, num_heads=nheads)
        # The prediction head for each masked token
        self.classifier_head = nn.Linear(dim_model, n_tokens+n_atokens, bias=False) 
        bias = nn.Parameter(torch.zeros(n_tokens+n_atokens)) # each token has its own bias
        self.classifier_head.bias =  bias
        self.activation = nn.Tanh()
        self.embeddings = nn.Embedding(num_embeddings=n_tokens+n_atokens, embedding_dim=dim_model, padding_idx=0)

        if pool == 'cls':
            context_length += 1
        

    def forward(self, x, attention_mask, **kwargs):
        token_embedding = self.embeddings(x) # batch x (context_length) x dim_model
        transformer_output, attn_scores = self.encoder(token_embedding, attention_mask) # batch x (n_tokens) x dim_model
        return {'transformer_output': transformer_output,
                'attention_score': attn_scores}
    

