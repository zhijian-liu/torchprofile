# Torchprofile

This is a profiler to count the number of MACs / FLOPs of PyTorch models based on `torch.jit.trace`.
* It is more **general** than ONNX-based profilers as some operations in PyTorch are not supported by ONNX for now.
* It is more **accurate** than hook-based profilers as they cannot profile operations within `torch.nn.Module`.

## Installation


```bash
pip install torchprofile
```

## Getting Started

You should first define your PyTorch model and its (dummy) input:

```python
import torch
from torchvision.models import resnet18

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

You can then measure the number of MACs using `profile_macs`:

```python
from torchprofile import profile_macs

macs = profile_macs(model, inputs)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
