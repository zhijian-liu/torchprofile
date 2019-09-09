import torch
from torch.nn.modules.transformer import Transformer

from torchprofile import profile_macs

if __name__ == '__main__':
    embed_size = 512
    num_tokens = 30

    model = Transformer(embed_size)
    source = torch.zeros(num_tokens, 1, embed_size)
    target = torch.zeros(num_tokens, 1, embed_size)

    total_macs = profile_macs(model, source, target)
    print(total_macs / 1e9, 'GFLOPs')
