# torchprofile

This tool is used to profile the NN models.

## Installation

We recommend you to install the latest version of this package from GitHub:

```bash
pip install --upgrade git+https://github.com/mit-han-lab/torchprofiler.git
```

## Getting Started

Before profiling, you should first define your PyTorch model and a (dummy) input which the model takes:

```python
from torchvision.models import resnet18
import torch

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

If you want to profile the number of multiplications in your model,

```python
from torchprofile import profile_macs
import numpy as np

macs = profile_macs(model, inputs)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.