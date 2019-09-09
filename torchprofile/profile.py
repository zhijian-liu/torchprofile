import warnings

import numpy as np

from .utils.trace import trace

__all__ = ['profile_macs']


def mul(node):
    # float[4, 4] = aten::mul(%0: float[4, 1], %1: float[1, 4])
    # float[n, m] = aten::mul(%0: float[n, m], %4: long[])
    return np.prod(node.outputs[0].shape)


def matmul(node):
    # float[16, 512], %4: float[512, 512]
    return np.prod(node.inputs[0].shape + [node.inputs[1].shape[-1]])


def bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b = node.outputs[0].shape[0]
    n = node.outputs[0].shape[1]
    p = node.outputs[0].shape[2]
    m = node.inputs[0].shape[2]
    return b * n * p * m


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


def convolution(node):
    return np.prod(node.outputs[0].shape + node.inputs[1].shape[1:])


def layer_norm(node):
    # [30, 1, 512] = aten::layer_norm(float[30, 1, 512], *, float[512], float[512], *, *)
    return np.prod(node.outputs[0].shape)


def mean(node):
    return 1


def zero(node):
    return 0


_handlers = (
    (('aten::add', 'aten::add_'), zero),
    (('aten::div', 'aten::div_'), zero),

    (('aten::mul', 'aten::mul_'), mul),
    ('aten::matmul', matmul),
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),

    ('aten::bmm', bmm),

    ('aten::_convolution', convolution),
    ('aten::layer_norm', layer_norm),

    ('aten::max_pool2d', zero),
    ('aten::mean', mean),

    # TODO: need to be fixed
    ('aten::batch_norm', zero),
    (('aten::avg_pool2d', 'aten::adaptive_avg_pool2d'), zero),

    (('aten::cat', 'aten::chunk', 'aten::clone', 'aten::contiguous', 'aten::dropout', 'aten::dropout_', 'aten::eq',
      'aten::flatten', 'aten::hardtanh_', 'aten::int', 'aten::ne', 'aten::relu', 'aten::relu_', 'aten::select',
      'aten::size', 'aten::slice', 'aten::softmax', 'aten::sum', 'aten::t', 'aten::transpose', 'aten::view',
      'prim::constant', 'prim::listconstruct', 'prim::listunpack', 'prim::numtotensor'), zero)
)


def profile_macs(model, *args, reduction=sum, **kwargs):
    graph = trace(model, *args, **kwargs)

    results = dict()
    for node in graph.nodes:
        for operator, func in _handlers:
            if node.operator == operator or (isinstance(operator, (list, tuple)) and node.operator in operator):
                res = func(node)
                results[node] = res
                break

        if node not in results:
            warnings.warn('missing handler for {}'.format(node.operator), UserWarning)

    if reduction is not None:
        return reduction(results.values())
    else:
        return results
