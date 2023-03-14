"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import math
import torch
import torch.nn as nn
# from .layers import LoRALayer

class LoRA(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim=128
    ):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(dim, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, dim))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

class Adapter(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        dim=128
    ):
        super().__init__()
        self.A = nn.Linear(in_features, dim)
        self.B = nn.Linear(dim, out_features)
        
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.zeros_(self.A.bias)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)
