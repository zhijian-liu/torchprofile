import torch
from torchvision import models

from torchprofile import profile_macs

if __name__ == '__main__':
    for name, model in models.__dict__.items():
        if name.islower() and not name.startswith('__') and 'inception' not in name and callable(model):
            model = model()
            inputs = torch.randn(1, 3, 224, 224)
            total_macs = profile_macs(model, inputs)
            print("%s: %.2f GMACs" % (name, total_macs / 1e9))
