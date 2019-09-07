import torch
from torch.nn.modules.transformer import Transformer

from torchprofile import profile_mults

transformer = Transformer()

src = torch.zeros(30, 1, 512)
tgt = torch.zeros(30, 1, 512)

total_ops = profile_mults(transformer, src, tgt)
print(sum(total_ops.values()) / 1e9)
