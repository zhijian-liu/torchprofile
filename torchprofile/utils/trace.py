import warnings

import torch
import torch.jit

from .flatten import Flatten
from .ir import Variable, Node, Graph

__all__ = ['trace']


def trace(model, args=(), kwargs=None):
    assert kwargs is None, 'Keyword arguments are not supported for now. ' \
                           'Please use positional arguments instead!'

    with warnings.catch_warnings(record=True):
        trace, _ = torch.jit.get_trace_graph(Flatten(model), args, kwargs)

    variables = dict()
    for node in trace.graph().nodes():
        for var in list(node.inputs()) + list(node.outputs()):
            if 'tensor' in var.type().kind().lower():
                variables[var] = Variable(name=var.debugName(),
                                          dtype=var.type().scalarType(),
                                          shape=var.type().sizes())
            else:
                variables[var] = Variable(name=var.debugName(),
                                          dtype=str(var.type()))

    nodes = []
    for node in trace.graph().nodes():
        node = Node(operator=node.kind(),
                    attributes={name: getattr(node, node.kindOf(name))(name) for name in node.attributeNames()},
                    inputs=[variables[var] for var in node.inputs()],
                    outputs=[variables[var] for var in node.outputs()],
                    scope=node.scopeName().replace('Flatten/', '', 1).replace('Flatten', '', 1))
        nodes.append(node)

    graph = Graph(name=model.__class__.__module__ + '.' + model.__class__.__name__,
                  variables=[var for var in variables.values()],
                  inputs=[variables[var] for var in trace.graph().inputs()],
                  outputs=[variables[var] for var in trace.graph().outputs()],
                  nodes=nodes)
    return graph
