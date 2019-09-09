# torchprofile

This tool is used to profile the MACs of PyTorch models. It is based on `torch.jit.trace`, which is more general and accurate than hook-based profilers.

## Installation

We recommend you to install the latest version of this package from GitHub:

```bash
pip install --upgrade git+https://github.com/mit-han-lab/torchprofile.git
```

To install from source code
```bash
git clone https://github.com/mit-han-lab/torchprofile.git
cd ./torchprofile
pip install .
```

## Getting Started

Before profiling, you should first define your PyTorch model and a (dummy) input which the model takes:

```python
from torchvision.models import resnet18
import torch

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

If you want to profile the number of MACs in your model,

```python
from torchprofile import profile_macs

macs = profile_macs(model, inputs)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
