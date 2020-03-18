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
        graph, _ = torch.jit._get_trace_graph(Flatten(model), args, kwargs)

    variables = dict()
    for node in graph.nodes():
        for v in list(node.inputs()) + list(node.outputs()):
            if 'tensor' in v.type().kind().lower():
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=v.type().scalarType(),
                    shape=v.type().sizes(),
                )
            else:
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=str(v.type()),
                )

    nodes = []
    for node in graph.nodes():
        node = Node(
            operator=node.kind(),
            attributes={
                x: getattr(node, node.kindOf(x))(x)
                for x in node.attributeNames()
            },
            inputs=[variables[v] for v in node.inputs() if v in variables],
            outputs=[variables[v] for v in node.outputs() if v in variables],
            scope=node.scopeName() \
                    .replace('Flatten/', '', 1) \
                    .replace('Flatten', '', 1),
        )
        nodes.append(node)

    graph = Graph(
        name=model.__class__.__module__ + '.' + model.__class__.__name__,
        variables=[v for v in variables.values()],
        inputs=[variables[v] for v in graph.inputs() if v in variables],
        outputs=[variables[v] for v in graph.outputs() if v in variables],
        nodes=nodes,
    )
    return graph
