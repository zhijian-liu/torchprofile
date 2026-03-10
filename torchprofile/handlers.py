from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir import Node


def _addmm(node: Node) -> int:
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    return n * m * p


def _addmv(node: Node) -> int:
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n, m = node.inputs[1].shape
    return n * m


def _bmm(node: Node) -> int:
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    return b * n * m * p


def _baddbmm(node: Node) -> int:
    # [b, n, p] = aten::baddbmm([b, n, p], [b, n, m], [b, m, p])
    b, n, p = node.inputs[0].shape
    b, n1, m = node.inputs[1].shape
    b, m1, p1 = node.inputs[2].shape
    assert n == n1 and m == m1 and p == p1
    return b * n * m * p


def _matmul(node: Node) -> int:
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        return n
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        n, m = node.inputs[1].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        n, m = node.inputs[0].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        return n * m * p
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        return prod(b) * n * m
    elif node.inputs[1].ndim == 1:
        # [..., n] = aten::matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        return prod(b) * n * m
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        return prod(b) * n * m * p


def _mul(node: Node) -> int:
    return prod(node.outputs[0].shape)


def _convolution(node: Node) -> int:
    output_shape = node.outputs[0].shape
    weight_shape = node.inputs[1].shape
    if output_shape[1] == weight_shape[0]:
        _, in_channels, *kernel_size = weight_shape
    else:
        in_channels, _, *kernel_size = weight_shape
    return prod(output_shape) * in_channels * prod(kernel_size)


def _norm(node: Node) -> int:
    if node.operator in ("aten::batch_norm", "aten::instance_norm"):
        affine = node.inputs[1].shape is not None
    elif node.operator in ("aten::layer_norm", "aten::group_norm"):
        affine = node.inputs[2].shape is not None
    else:
        raise ValueError(node.operator)
    return prod(node.outputs[0].shape) if affine else 0


def _avg_pool_or_mean(node: Node) -> int:
    return prod(node.outputs[0].shape)


def _leaky_relu(node: Node) -> int:
    return prod(node.outputs[0].shape)


def _upsample_bilinear2d(node: Node) -> int:
    return prod(node.outputs[0].shape) * 4


def _lstm(node: Node) -> int:
    # input[0]: (seq_len, batch, input_size) or (batch, seq_len, input_size)
    # output[0]: (*, num_directions * hidden_size)
    # output[1] (h_n): (num_layers * num_directions, batch, hidden_size)
    input_size = node.inputs[0].shape[-1]
    hidden_size = node.outputs[1].shape[-1]
    num_directions = node.outputs[0].shape[-1] // hidden_size
    num_layers = node.outputs[1].shape[0] // num_directions
    batch_x_seq = node.outputs[0].shape[0] * node.outputs[0].shape[1]

    macs = 0
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        # 4 gates, each: input-hidden matmul + hidden-hidden matmul
        macs += 4 * hidden_size * (layer_input_size + hidden_size)

    return batch_x_seq * num_directions * macs


def _einsum(node: Node) -> int:
    equation = node.inputs[0].value
    tensor_shapes = node.inputs[1].value
    if not equation or not tensor_shapes or "..." in equation:
        return 0

    input_part = equation.split("->")[0]
    input_subs = input_part.split(",")

    index_sizes = {}
    for sub, shape in zip(input_subs, tensor_shapes):
        if shape is None:
            return 0
        for letter, size in zip(sub.replace(" ", ""), shape):
            index_sizes[letter] = size

    return prod(index_sizes.values())


def _grid_sampler(node: Node) -> int:
    # Bilinear: 4 MACs, nearest: 0, bicubic: 16 MACs per output element
    output_size = prod(node.outputs[0].shape)
    mode = node.inputs[2].value
    if mode == 1:  # nearest
        return 0
    if mode == 2:  # bicubic
        return output_size * 16
    return output_size * 4  # bilinear (default)


def _scaled_dot_product_attention(node: Node) -> int:
    # Q: (batch, *heads, seq_q, head_dim)
    # K: (batch, *heads, seq_kv, head_dim)
    # V: (batch, *heads, seq_kv, head_dim_v)
    q, k, v = node.inputs[0].shape, node.inputs[1].shape, node.inputs[2].shape

    batch = q[0]
    heads = prod(q[1:-2])
    seq_q = q[-2]
    head_dim = q[-1]
    seq_kv = k[-2]
    head_dim_v = v[-1]

    assert batch == k[0] == v[0]
    assert heads == prod(k[1:-2]) == prod(v[1:-2])
    assert head_dim == k[-1] and seq_kv == v[-2]

    # Q @ K^T + attn @ V
    return batch * heads * seq_q * seq_kv * (head_dim + head_dim_v)


# Operator -> handler function. None means recognized but zero MACs.
HANDLER_MAP = {
    "aten::scaled_dot_product_attention": _scaled_dot_product_attention,
    "aten::lstm": _lstm,
    "aten::einsum": _einsum,
    "aten::grid_sampler": _grid_sampler,
    "aten::addmm": _addmm,
    "aten::addmv": _addmv,
    "aten::bmm": _bmm,
    "aten::baddbmm": _baddbmm,
    "aten::linear": _matmul,
    "aten::matmul": _matmul,
    "aten::mul": _mul,
    "aten::mul_": _mul,
    "aten::_convolution": _convolution,
    "aten::batch_norm": _norm,
    "aten::instance_norm": _norm,
    "aten::layer_norm": _norm,
    "aten::group_norm": _norm,
    "aten::adaptive_avg_pool1d": _avg_pool_or_mean,
    "aten::adaptive_avg_pool2d": _avg_pool_or_mean,
    "aten::adaptive_avg_pool3d": _avg_pool_or_mean,
    "aten::avg_pool1d": _avg_pool_or_mean,
    "aten::avg_pool2d": _avg_pool_or_mean,
    "aten::avg_pool3d": _avg_pool_or_mean,
    "aten::mean": _avg_pool_or_mean,
    "aten::leaky_relu": _leaky_relu,
    "aten::upsample_bilinear2d": _upsample_bilinear2d,
}

_ZERO_COST_OPS = (
    "aten::abs",
    "aten::adaptive_max_pool1d",
    "aten::adaptive_max_pool2d",
    "aten::adaptive_max_pool3d",
    "aten::add",
    "aten::add_",
    "aten::alpha_dropout",
    "aten::arange",
    "aten::cat",
    "aten::chunk",
    "aten::clamp",
    "aten::clone",
    "aten::constant_pad_nd",
    "aten::contiguous",
    "aten::copy_",
    "aten::cos",
    "aten::detach",
    "aten::div",
    "aten::div_",
    "aten::dropout",
    "aten::dropout_",
    "aten::embedding",
    "aten::eq",
    "aten::expand",
    "aten::feature_dropout",
    "aten::fft_fft",
    "aten::flatten",
    "aten::floor",
    "aten::floor_divide",
    "aten::gather",
    "aten::ge",
    "aten::gelu",
    "aten::gt",
    "aten::hardtanh",
    "aten::hardtanh_",
    "aten::imag",
    "aten::index",
    "aten::int",
    "aten::le",
    "aten::log2",
    "aten::log_softmax",
    "aten::lt",
    "aten::max",
    "aten::max_pool1d",
    "aten::max_pool1d_with_indices",
    "aten::max_pool2d",
    "aten::max_pool2d_with_indices",
    "aten::max_pool3d",
    "aten::max_pool3d_with_indices",
    "aten::max_unpool1d",
    "aten::max_unpool2d",
    "aten::max_unpool3d",
    "aten::ne",
    "aten::neg",
    "aten::permute",
    "aten::pow",
    "aten::real",
    "aten::reciprocal",
    "aten::reflection_pad1d",
    "aten::reflection_pad2d",
    "aten::reflection_pad3d",
    "aten::relu",
    "aten::relu_",
    "aten::replication_pad1d",
    "aten::replication_pad2d",
    "aten::replication_pad3d",
    "aten::reshape",
    "aten::rsub",
    "aten::rsqrt",
    "aten::scalarimplicit",
    "aten::select",
    "aten::sigmoid",
    "aten::sign",
    "aten::silu",
    "aten::sin",
    "aten::size",
    "aten::slice",
    "aten::softmax",
    "aten::softshrink",
    "aten::split",
    "aten::split_with_sizes",
    "aten::sqrt",
    "aten::squeeze",
    "aten::stack",
    "aten::sub",
    "aten::sum",
    "aten::t",
    "aten::tanh",
    "aten::threshold",
    "aten::to",
    "aten::transpose",
    "aten::unflatten",
    "aten::unfold",
    "aten::unsqueeze",
    "aten::upsample_nearest2d",
    "aten::view",
    "aten::zeros",
    "prim::constant",
    "prim::listconstruct",
    "prim::listunpack",
    "prim::numtotensor",
    "prim::tupleconstruct",
)

HANDLER_MAP.update(dict.fromkeys(_ZERO_COST_OPS))
