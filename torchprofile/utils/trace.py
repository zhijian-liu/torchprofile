import warnings

import torch
import torch.jit

from .flatten import Flatten
from .ir import Graph, Node, Variable

__all__ = ['trace']


def trace(model, args=(), kwargs=None):
    assert kwargs is None, 'Keyword arguments are not supported for now. ' \
                           'Please use positional arguments instead!'

    with warnings.catch_warnings(record=True):
        graph, _ = torch.jit._get_trace_graph(Flatten(model), args, kwargs)

    variables = dict()
    for x in graph.nodes():
        for v in list(x.inputs()) + list(x.outputs()):
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
    for x in graph.nodes():
        node = Node(
            operator=x.kind(),
            attributes={
                s: getattr(x, x.kindOf(s))(s)
                for s in x.attributeNames()
            },
            inputs=[variables[v] for v in x.inputs() if v in variables],
            outputs=[variables[v] for v in x.outputs() if v in variables],
            scope=x.scopeName() \
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
