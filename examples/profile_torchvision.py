import torch
from torchvision import models

from torchprofile import profile_macs

if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__ if
                         name.islower() and not name.startswith('__') and 'inception' not in name
                         and callable(models.__dict__[name]))

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_names = ['mobilenet_v2']

    for name in model_names:
        model = models.__dict__[name]().to(device)
        inputs = torch.randn((1, 3, 224, 224)).to(device)
        total_ops = profile_macs(model, inputs)
        print("%s: %.2f GFLOPs" % (name, total_ops / 1e9))
