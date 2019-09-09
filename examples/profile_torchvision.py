import torch
from torchvision import models

from torchprofile import profile_macs

if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__ if
                         name.islower() and not name.startswith('__') and 'inception' not in name
                         and callable(models.__dict__[name]))

    for name in model_names:
        model = models.__dict__[name]()
        inputs = torch.randn(1, 3, 224, 224)
        total_macs = profile_macs(model, inputs)
        print("%s: %.2f GMACs" % (name, total_macs / 1e9))
