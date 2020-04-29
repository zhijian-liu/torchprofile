import torch
import torch.nn as nn
from torchprofile.utils.trace import trace

if __name__ == '__main__':
    in_features = 16
    out_features = 32

    model = nn.Linear(in_features, out_features)
    inputs = torch.randn(1, in_features)

    graph = trace(model, inputs)
    print(graph)
