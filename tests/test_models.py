"""Tests for end-to-end profiling of real models."""

import pytest
import torch

from torchprofile import profile_macs


def _make_gpt2():
    from transformers import GPT2Config, GPT2Model

    return GPT2Model(GPT2Config()).eval()


def _make_bert():
    from transformers import BertConfig, BertModel

    return BertModel(BertConfig()).eval()


def _make_llama():
    from transformers import LlamaConfig, LlamaModel

    return LlamaModel(
        LlamaConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            max_position_embeddings=256,
        )
    ).eval()


def _make_resnet18():
    from torchvision.models import resnet18

    return resnet18().eval()


def _make_mobilenet_v2():
    from torchvision.models import mobilenet_v2

    return mobilenet_v2().eval()


def _make_vit_b_16():
    from torchvision.models import vit_b_16

    return vit_b_16().eval()


MODEL_CASES = [
    ("gpt2", _make_gpt2, torch.randint(0, 50257, (1, 128)), 11.195, 0.01),
    ("bert", _make_bert, torch.randint(0, 30522, (1, 128)), 11.177, 0.01),
    ("llama", _make_llama, torch.randint(0, 1000, (1, 64)), 0.622, 0.01),
    ("resnet18", _make_resnet18, torch.randn(1, 3, 224, 224), 1.817, 0.01),
    ("mobilenet_v2", _make_mobilenet_v2, torch.randn(1, 3, 224, 224), 0.3075, 0.01),
    ("vit_b_16", _make_vit_b_16, torch.randn(1, 3, 224, 224), 17.57, 0.1),
]


@pytest.mark.parametrize(
    "factory, inputs, expected_gmacs, tol",
    [(f, i, g, t) for _, f, i, g, t in MODEL_CASES],
    ids=[id for id, *_ in MODEL_CASES],
)
def test_model(factory, inputs, expected_gmacs, tol):
    macs = profile_macs(factory(), inputs)
    assert abs(macs / 1e9 - expected_gmacs) < tol
