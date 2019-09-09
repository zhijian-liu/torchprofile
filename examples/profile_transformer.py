import torch
from torch.nn.modules.transformer import Transformer

from torchprofile import profile_macs

if __name__ == '__main__':
    transformer = Transformer()
    source = torch.zeros(30, 1, 512)
    target = torch.zeros(30, 1, 512)

    total_ops = profile_macs(transformer, source, target)
    print(total_ops / 1e9)
