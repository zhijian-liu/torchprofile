"""Tests for individual operator handlers with exact MAC counts."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchprofile import profile_macs

# ── Helpers ──────────────────────────────────────────────────────────────


class Lambda(nn.Module):
    """Wrap an arbitrary callable as an nn.Module for profiling."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


# ── Parametrized operator tests ──────────────────────────────────────────

OPERATOR_CASES = [
    # (id, model, inputs, expected_macs)
    # --- linear / matmul ---
    ("linear_16x32", nn.Linear(16, 32, bias=True), torch.randn(1, 16), 1 * 16 * 32),
    ("linear_batched", nn.Linear(16, 32, bias=True), torch.randn(4, 16), 4 * 16 * 32),
    ("matmul_mv", Lambda(lambda a, b: a @ b), (torch.randn(3, 4), torch.randn(4)), 3 * 4),
    ("matmul_mm", Lambda(lambda a, b: a @ b), (torch.randn(3, 4), torch.randn(4, 5)), 3 * 4 * 5),
    ("bmm", Lambda(lambda a, b: torch.bmm(a, b)), (torch.randn(2, 3, 4), torch.randn(2, 4, 5)), 2 * 3 * 4 * 5),
    # --- convolution ---
    (
        "conv1d_k3",
        nn.Conv1d(3, 16, kernel_size=3, bias=False),
        torch.randn(1, 3, 32),
        1 * 16 * 30 * 3 * 3,
    ),  # output_len = 30
    ("conv2d_k3_pad1", nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False), torch.randn(1, 3, 8, 8), 27648),
    (
        "depthwise_conv2d",
        nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
        torch.randn(1, 16, 8, 8),
        1 * 16 * 8 * 8 * 1 * 3 * 3,
    ),
    # --- normalization ---
    ("batch_norm", nn.BatchNorm2d(16), torch.randn(1, 16, 8, 8), 1 * 16 * 8 * 8),
    ("layer_norm", nn.LayerNorm(64), torch.randn(1, 10, 64), 1 * 10 * 64),
    ("group_norm", nn.GroupNorm(4, 16), torch.randn(1, 16, 8, 8), 1 * 16 * 8 * 8),
    # --- pooling ---
    ("adaptive_avg_pool2d", nn.AdaptiveAvgPool2d(1), torch.randn(1, 16, 8, 8), 1 * 16),
    ("avg_pool2d", nn.AvgPool2d(2), torch.randn(1, 16, 8, 8), 1 * 16 * 4 * 4),
    # --- recurrent ---
    ("lstm_1layer", nn.LSTM(input_size=16, hidden_size=32, num_layers=1), torch.randn(5, 1, 16), 30720),
    (
        "lstm_2layer_bidir",
        nn.LSTM(input_size=32, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True),
        torch.randn(1, 10, 32),
        1474560,
    ),
    # --- einsum ---
    (
        "einsum_bmm",
        Lambda(lambda a, b: torch.einsum("bij,bjk->bik", a, b)),
        (torch.randn(2, 3, 4), torch.randn(2, 4, 5)),
        120,
    ),
    (
        "einsum_attention",
        Lambda(lambda q, k: torch.einsum("bhqd,bhkd->bhqk", q, k)),
        (torch.randn(1, 8, 16, 64), torch.randn(1, 8, 16, 64)),
        1 * 8 * 16 * 64 * 16,
    ),
    # --- grid sampler ---
    (
        "grid_sample_bilinear",
        Lambda(lambda x, g: F.grid_sample(x, g, mode="bilinear", align_corners=True)),
        (torch.randn(1, 3, 8, 8), torch.randn(1, 4, 4, 2)),
        192,
    ),
    (
        "grid_sample_nearest",
        Lambda(lambda x, g: F.grid_sample(x, g, mode="nearest", align_corners=True)),
        (torch.randn(1, 3, 8, 8), torch.randn(1, 4, 4, 2)),
        0,
    ),
    # --- activation ---
    ("leaky_relu", nn.LeakyReLU(), torch.randn(1, 16, 8, 8), 1 * 16 * 8 * 8),
    ("relu_zero_cost", nn.ReLU(), torch.randn(1, 16, 8, 8), 0),
]


@pytest.mark.parametrize(
    "model, inputs, expected_macs",
    [(m, i, e) for _, m, i, e in OPERATOR_CASES],
    ids=[id for id, *_ in OPERATOR_CASES],
)
def test_operator(model, inputs, expected_macs):
    model.eval()
    macs = profile_macs(model, inputs)
    assert macs == expected_macs


def test_reduction_none():
    model = nn.Linear(16, 32).eval()
    results = profile_macs(model, torch.randn(1, 16), reduction=None)
    assert isinstance(results, dict)
    assert sum(results.values()) == 512
