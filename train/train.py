from pathlib import Path
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from datasets import load_from_disk
from datasets import DatasetDict, load_dataset, concatenate_datasets
import signal
import torch
import json
import numpy as np
import logging
#setting the python environment
import sys
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(cur_path))
from scCLIP.utils import *
from scCLIP.model import *

#setting the wandb environment
os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001820/scCLIP/scCLIP/cache"
os.environ["WANDB_DIR"] = "/scratch/project_465001820/scCLIP/scCLIP/cache"
os.environ["WANDB_CONFIG_DIR"] = "/scratch/project_465001820/scCLIP/scCLIP/cache"
os.environ["WANDB_CACHE_DIR"] = "/scratch/project_465001820/scCLIP/scCLIP/cache"
hf_cache = "/scratch/project_465001820/scCLIP/scCLIP/cache"


#setting the running device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#track the NAN loss
torch.autograd.set_detect_anomaly(True)

def manual_train_fm(config=None):
    
    pl.seed_everything(42)
    # import pdb; pdb.set_trace()
    model = Spaformer(dim_model=config['dim_model'], 
                        nheads=config['nheads'], 
                        nlayers=config['nlayers'],
                        dropout=config['dropout'],
                        masking_p=config['masking_p'], 
                        n_tokens=config['n_tokens'],
                        n_atokens=config['n_atokens'],
                        context_length=config['context_length'],
                        lr=config['lr'],
                        warmup=config['warmup'],
                        max_epochs=config['max_epochs'],
                        pool=config['pool'],
                        outer_config = config)
                      
    return model


if __name__ == "__main__":
    with open(os.path.join("/scratch/project_465001820/scCLIP/scCLIP/config/_config_train.json"), 'r') as json_file:
        config = json.load(json_file)
    
    meta_counter = int(config["organ"]) + int(config["specie"]) + int(config["assay"]) + int(config["condition"])


    plmodel = manual_train_fm(config=config)
    
    x = torch.randint(1, 100, (1, 100))
    adjmtx = None
    attention_mask = torch.ones(1, 100) == 1
    output = plmodel(x, attention_mask)
    import pdb; pdb.set_trace()