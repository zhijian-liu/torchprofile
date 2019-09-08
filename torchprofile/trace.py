import warnings

import torch
import torch.jit

from .ir import Tensor, Node, Graph
from .utils import Flatten

__all__ = ['trace']


def trace(model, *args, **kwargs):
    assert not kwargs, 'Keyword arguments are not supported for now. ' \
                       'Please use positional arguments instead!'

    with warnings.catch_warnings(record=True):
        trace, _ = torch.jit.get_trace_graph(Flatten(model), tuple(args), kwargs=kwargs)

    tensors = dict()
    for node in trace.graph().nodes():
        for var in list(node.inputs()) + list(node.outputs()):
            if 'tensor' in var.type().kind().lower():
                dtype = var.type().scalarType()
                shape = var.type().sizes()
            else:
                dtype = str(var.type())
                shape = None
            tensors[var] = Tensor(name=var.debugName(), dtype=dtype, shape=shape)

    nodes = []
    for node in trace.graph().nodes():
        attributes = {name: getattr(node, node.kindOf(name))(name) for name in node.attributeNames()}
        inputs = [tensors[var] for var in node.inputs()]
        outputs = [tensors[var] for var in node.outputs()]
        scope = node.scopeName().replace('Flatten/', '', 1).replace('Flatten', '', 1)
        nodes.append(Node(operator=node.kind(), attributes=attributes, inputs=inputs, outputs=outputs, scope=scope))

    inputs = [tensors[var] for var in trace.graph().inputs()]
    outputs = [tensors[var] for var in trace.graph().outputs()]
    return Graph(tensors=tensors.values(), inputs=inputs, outputs=outputs, nodes=nodes)
