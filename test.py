import torch
import torch.nn as nn
from torchprofile import profile_macs
from torchprofile.utils.trace import trace


class Model(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


if __name__ == '__main__':
    a = torch.zeros(10, 20, 1, 20, 20)
    b = torch.zeros(20, 30)

    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(trace(rnn, (input, (h0, c0))))
    print(profile_macs(rnn, (input, (h0, c0))))
