# torchprofile

This is a `torch.jit.trace`-based profiler to count the number of MACs / FLOPs for PyTorch models.
* It is more **general** than ONNX-based profilers as some operations in PyTorch are not supported by ONNX for now.
* It is more **accurate** than hook-based profilers as they cannot profile operations within the `torch.nn.Module`.

## Installation

We recommend you to install the latest version of this package from GitHub:

```bash
pip install --upgrade git+https://github.com/mit-han-lab/torchprofile.git
```

## Getting Started

Before profiling, you should first define your PyTorch model and a (dummy) input which the model takes:

```python
import torch
from torchvision.models import resnet18

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

If you want to profile the number of MACs in your model,

```python
from torchprofile import profile_macs

macs = profile_macs(model, (inputs,))
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.