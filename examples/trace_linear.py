import torch
import torch.nn as nn

from torchprofile.utils.trace import trace

if __name__ == '__main__':
    model = nn.Linear(16, 32)
    inputs = torch.zeros(1, 16)
    graph = trace(model, inputs)
    print(graph)
