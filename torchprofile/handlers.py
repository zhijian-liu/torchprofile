import numpy as np

__all__ = ['handlers']


def addmm(node):
    # [n, p] = aten::addmm([n(, p)], [n, m], [m, p], *, *)
    n = node.outputs[0].shape[0]
    p = node.outputs[0].shape[1]
    m = node.inputs[1].shape[1]
    return n * p * m


def addmv(node):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n = node.outputs[0].shape[0]
    m = node.inputs[1].shape[1]
    return n * m


def bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b = node.outputs[0].shape[0]
    n = node.outputs[0].shape[1]
    p = node.outputs[0].shape[2]
    m = node.inputs[0].shape[2]
    return b * n * p * m


def matmul(node):
    # float[16, 512], %4: float[512, 512]
    return np.prod(node.inputs[0].shape + [node.inputs[1].shape[-1]])


def mul(node):
    # float[4, 4] = aten::mul(%0: float[4, 1], %1: float[1, 4])
    # float[n, m] = aten::mul(%0: float[n, m], %4: long[])
    return np.prod(node.outputs[0].shape)


def convolution(node):
    return np.prod(node.outputs[0].shape + node.inputs[1].shape[1:])


def batch_norm(node):
    return 0


def layer_norm(node):
    # [30, 1, 512] = aten::layer_norm(float[30, 1, 512], *, float[512], float[512], *, *)
    return np.prod(node.outputs[0].shape)


def mean(node):
    return 1


def zero(node):
    return 0


handlers = (
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),
    ('aten::bmm', bmm),
    ('aten::matmul', matmul),
    (('aten::mul', 'aten::mul_'), mul),

    ('aten::_convolution', convolution),

    ('aten::batch_norm', batch_norm),
    ('aten::layer_norm', layer_norm),

    ('aten::mean', mean),

    # TODO: need to be fixed
    ('aten::avg_pool2d', zero),
    ('aten::adaptive_avg_pool2d', zero),

    (('aten::add', 'aten::add_', 'aten::cat', 'aten::chunk', 'aten::clone', 'aten::contiguous', 'aten::div',
      'aten::div_' 'aten::dropout', 'aten::dropout_', 'aten::eq', 'aten::flatten', 'aten::hardtanh_', 'aten::int',
      'aten::max_pool1d', 'aten::max_pool2d', 'aten::max_pool3d', 'aten::ne', 'aten::relu', 'aten::relu_',
      'aten::select', 'aten::size', 'aten::slice', 'aten::softmax', 'aten::sum', 'aten::t', 'aten::transpose',
      'aten::view', 'prim::constant', 'prim::listconstruct', 'prim::listunpack', 'prim::numtotensor'), zero)
)
