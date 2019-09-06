import torch
from torchvision import models

from torchprofile import profile_macs

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__") and not "inception" in name
                     and callable(models.__dict__[name]))

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

for name in model_names:
    model = models.__dict__[name]().to(device)
    inputs = torch.randn((1, 3, 224, 224)).to(device)
    total_ops = profile_macs(model, inputs)
    print("%s: %.2f GFLOPs" % (name, sum(total_ops.values()) / 1e9))
