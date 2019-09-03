import numpy as np

from .trace import trace_graph

__all__ = ['profile_macs']

_handlers = (
    ('aten::_convolution', lambda node: np.prod(node.outputs[0].shape + node.inputs[1].shape[1:])),
    ('aten::addmm', lambda node: np.prod(node.outputs[0].shape + [node.inputs[2].shape[0]])),
    ('aten::matmul', lambda node: np.prod(node.inputs[0].shape + [node.inputs[1].shape[-1]])),
    ('aten::bmm', lambda node: np.prod(node.inputs[0].shape + [node.inputs[1].shape[-1]])),
    ('aten::mean', lambda node: 1)
)

SKIP = [
    'prim::constant', 'prim::listconstruct', 'prim::numtotensor',
    'aten::clone', 'aten::dropout', 'aten::eq', 'aten::size', 'aten::t'
]


def profile_macs(model, *args, **kwargs):
    nodes = trace_graph(model, *args, **kwargs)

    macs = dict()
    for node in nodes:
        found = False
        for operator, func in _handlers:
            if node.operator == operator:
                macs[node] = func(node)
                found = True
                break

        # if not found and node.operator not in SKIP:
        #     print(node.operator)
    return macs
