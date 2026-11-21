"""Bias-free residual bottleneck; initialization never consumes CE RNG."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBottleneck(nn.Module):
    def __init__(self, width, rank, generator):
        super().__init__()
        self.down = nn.Parameter(torch.empty(rank, width))
        self.up = nn.Parameter(torch.zeros(width, rank))
        nn.init.uniform_(self.down, -1 / math.sqrt(width), 1 / math.sqrt(width),
                         generator=generator)

    def forward(self, value):
        normalized = value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        return F.linear(F.gelu(F.linear(normalized, self.down)), self.up)
