# TorchProfile

[![PyPI](https://img.shields.io/pypi/v/torchprofile)](https://pypi.org/project/torchprofile/)
[![License](https://img.shields.io/github/license/zhijian-liu/torchprofile)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c)](https://pytorch.org/)

TorchProfile counts the number of MACs (multiply-accumulate operations) in a PyTorch model. It works by tracing the computation graph with `torch.jit.trace`, making it more accurate than hook-based profilers and more general than ONNX-based ones.

## Installation

```bash
pip install torchprofile
```

## Quick Start

```python
import torch
from transformers import AutoModel
from torchprofile import profile_macs

model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B").eval()
inputs = torch.randint(0, model.config.vocab_size, (1, 128))

macs = profile_macs(model, inputs)
print(f"{macs / 1e9:.2f} GMACs")
```

To get a per-operator breakdown, pass `reduction=None`:

```python
results = profile_macs(model, inputs, reduction=None)
for node, macs in results.items():
    if macs > 0:
        print(f"{node.scope:40s} {node.operator:30s} {macs / 1e6:>8.2f} MMACs")
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
