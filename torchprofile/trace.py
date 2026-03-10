from __future__ import annotations

import warnings
from collections import deque
from typing import Any

import torch
import torch.jit
import torch.nn as nn

from .ir import Graph, Node, Variable


def _flatten(inputs: Any) -> list[torch.Tensor]:
    """Recursively unpack nested lists/tuples/dicts into a flat list of tensors."""
    queue = deque([inputs])
    outputs: list[torch.Tensor] = []
    while queue:
        x = queue.popleft()
        if isinstance(x, (list, tuple)):
            queue.extend(x)
        elif isinstance(x, dict):
            queue.extend(x.values())
        elif isinstance(x, torch.Tensor):
            outputs.append(x)
    return outputs


class _Flatten(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]:
        return _flatten(self.model(*args, **kwargs))


def _extract_value(v: Any) -> Any:
    """Extract the runtime value of a non-tensor JIT value when possible."""
    try:
        producer = v.node()
        if producer.kind() == "prim::Constant":
            attrs = list(producer.attributeNames())
            if attrs:
                return getattr(producer, producer.kindOf(attrs[0]))(attrs[0])
        elif producer.kind() == "prim::ListConstruct":
            shapes = []
            for elem in producer.inputs():
                if "tensor" in elem.type().kind().lower():
                    shapes.append(elem.type().sizes())
                else:
                    shapes.append(None)
            if shapes:
                return shapes
    except Exception:
        pass
    return None


def trace(model: nn.Module, args: Any = (), kwargs: dict[str, Any] | None = None) -> Graph:
    """Trace a model and return an intermediate representation of its computation graph.

    Args:
        model: The PyTorch model to trace.
        args: Positional arguments (a single tensor or a tuple of tensors).
        kwargs: Keyword arguments (not yet supported, must be ``None``).

    Returns:
        A :class:`~torchprofile.ir.Graph` containing all operations and their
        tensor shapes.
    """
    if kwargs is not None:
        raise ValueError("Keyword arguments are not supported for now. Please use positional arguments instead.")

    with warnings.catch_warnings(record=True):
        jit_graph, _ = torch.jit._get_trace_graph(_Flatten(model), args, kwargs)

    variables: dict[Any, Variable] = {}
    for jit_node in jit_graph.nodes():
        for jit_val in list(jit_node.inputs()) + list(jit_node.outputs()):
            if "tensor" in jit_val.type().kind().lower():
                variables[jit_val] = Variable(
                    name=jit_val.debugName(),
                    dtype=jit_val.type().scalarType(),
                    shape=jit_val.type().sizes(),
                )
            else:
                variables[jit_val] = Variable(
                    name=jit_val.debugName(),
                    dtype=str(jit_val.type()),
                    value=_extract_value(jit_val),
                )

    nodes: list[Node] = []
    for jit_node in jit_graph.nodes():
        node = Node(
            operator=jit_node.kind(),
            attributes={s: getattr(jit_node, jit_node.kindOf(s))(s) for s in jit_node.attributeNames()},
            inputs=[variables[v] for v in jit_node.inputs() if v in variables],
            outputs=[variables[v] for v in jit_node.outputs() if v in variables],
            scope=jit_node.scopeName().replace("_Flatten/", "", 1).replace("_Flatten", "", 1),
        )
        nodes.append(node)

    return Graph(
        name=f"{model.__class__.__module__}.{model.__class__.__name__}",
        variables=list(variables.values()),
        inputs=[variables[v] for v in jit_graph.inputs() if v in variables],
        outputs=[variables[v] for v in jit_graph.outputs() if v in variables],
        nodes=nodes,
    )
