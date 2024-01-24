import torch 
import torch.nn as nn
import torch.nn.functional as F 
from dataclasses import dataclass
from typing import Optional

@dataclass
class Modelargs:
     dim: int = 4096
     n_layers: int = 32
     n_heads: int = 32
     n_kv_heads: Optional[int] = None
     ffn_dim_multiplier: Optional[float] = None
     vocab_size: int = -1
     multiple_of: int = 256
     norm_eps: float = 1e-5
     
     max_batch_size: int = 32
     max_seq_len: int = 2048
     
     
class Transformer(nn.Module):
    def __init__(self, args: Modelargs) -> None:
         super().__init__()
         
         self.args = args
         self.vocab_size = args.vocab_size
         self.n_layers = args.n_layers
         self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
         
         self.layers = nn.ModuleList()
         
         for _ in range(self.n_layers):
             self.layers.append(Encoderblock(args))
             
         self.norm = RMSNorm(args.dim , eps=args.norm_eps)
         self.output = dd
             
             
       
        
