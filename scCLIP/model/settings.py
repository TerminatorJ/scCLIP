import torch
from dataclasses import dataclass

@dataclass
class Settings:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.bfloat16
