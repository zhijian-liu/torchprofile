import numpy as np

from .trace import trace_graph

__all__ = ['profile_mults']

_handlers = (
    ('aten::_convolution', lambda node: np.prod(node.outputs[0].shape + node.inputs[1].shape[1:])),
    ('aten::addmm', lambda node: np.prod(node.outputs[0].shape + [node.inputs[2].shape[0]])),
    (('aten::matmul', 'aten::bmm'), lambda node: np.prod(node.inputs[0].shape + [node.inputs[1].shape[-1]])),
    ('aten::mean', lambda node: 1)
)

SKIP = [
    'prim::constant', 'prim::listconstruct', 'prim::numtotensor',
    'aten::clone', 'aten::dropout', 'aten::eq', 'aten::size', 'aten::t'
]


def profile_mults(model, *args, **kwargs):
    nodes = trace_graph(model, *args, **kwargs)

    macs = dict()
    for node in nodes:
        for operator, func in _handlers:
            if node.operator == operator or (isinstance(operator, (list, tuple)) and node.operator in operator):
                macs[node] = func(node)
                break

        # if not found and node.operator not in SKIP:
        #     print(node.operator)
    return macs
