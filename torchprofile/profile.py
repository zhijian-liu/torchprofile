import warnings

import numpy as np

from .utils.trace import trace

__all__ = ['profile', 'profile_flops', 'profile_macs']


class Result:
    def __init__(self, adds=0, mults=0, divs=0):
        self.adds = adds
        self.mults = mults
        self.divs = divs

    def __add__(self, other):
        self.adds += other.adds
        self.mults += other.mults
        self.divs += other.divs
        return self


def _add(node):
    return Result()


def _mul(node):
    assert len(node.outputs) == 1, node.outputs
    assert node.outputs[0].shape == node.inputs[0].shape, (node.outputs[0].shape, node.inputs[0].shape)
    return Result(mults=np.prod(node.outputs[0].shape))


def _div(node):
    assert len(node.outputs) == 1, node.outputs
    assert node.outputs[0].shape == node.inputs[0].shape, (node.outputs[0].shape, node.inputs[0].shape)
    return Result(divs=np.prod(node.outputs[0].shape))


def _convolution(node):
    assert len(node.outputs) == 1
    return Result(mults=np.prod(node.outputs[0].shape + node.inputs[1].shape[1:]))


def _addmm(node):
    return Result(mults=np.prod(node.outputs[0].shape + [node.inputs[2].shape[0]]))


def _matmul_or_bmm(node):
    return Result(mults=np.prod(node.inputs[0].shape + [node.inputs[1].shape[-1]]))


def _mean(node):
    return Result()


def _zero(node):
    return Result()


_handlers = (
    (('aten::add', 'aten::add_'), _add),
    (('aten::mul', 'aten::mul_'), _mul),
    (('aten::div', 'aten::div_'), _div),

    ('aten::addmm', _addmm),
    (('aten::matmul', 'aten::bmm'), _matmul_or_bmm),

    ('aten::_convolution', _convolution),
    ('aten::mean', _mean),

    (('aten::chunk', 'aten::clone', 'aten::contiguous', 'aten::dropout', 'aten::eq', 'aten::hardtanh_', 'aten::int',
      'aten::ne', 'aten::relu', 'aten::select', 'aten::size', 'aten::slice', 'aten::sum', 'aten::t', 'aten::transpose',
      'aten::view', 'prim::constant', 'prim::listconstruct', 'prim::listunpack', 'prim::numtotensor'), _zero)
)


def profile(model, *args, **kwargs):
    graph = trace(model, *args, **kwargs)

    total = Result()
    results = dict()
    for node in graph.nodes:
        for operator, func in _handlers:
            if node.operator == operator or (isinstance(operator, (list, tuple)) and node.operator in operator):
                res = func(node)
                results[node] = res
                total += res
                break

        if node not in results:
            warnings.warn('missing handler for {}'.format(node.operator), UserWarning)
            # print(node, node.scope)
    return total


def profile_flops(model, *args, **kwargs):
    return


def profile_macs(model, *args, **kwargs):
    results = profile(model, *args, **kwargs)
    return results.mults
