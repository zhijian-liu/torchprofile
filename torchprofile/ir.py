from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False, repr=False)
class Variable:
    """A tensor or scalar variable in the traced computation graph.

    Attributes:
        name: Debug name from the JIT trace (e.g. ``"%42"``).
        dtype: Scalar type (e.g. ``"float"``, ``"int"``).
        shape: Tensor dimensions, or ``None`` for non-tensor values.
        value: For constant scalars/strings, the runtime value extracted
            from the JIT graph.  ``None`` if unavailable or not a constant.
    """

    name: str
    dtype: str
    shape: list[int] | None = None
    value: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.dtype = self.dtype.lower()

    @property
    def ndim(self) -> int:
        """Number of dimensions (only valid when *shape* is not ``None``)."""
        return len(self.shape)

    def __repr__(self) -> str:
        text = f"%{self.name}: {self.dtype}"
        if self.shape is not None:
            text += "[" + ", ".join(str(x) for x in self.shape) + "]"
        return text


@dataclass(eq=False, repr=False)
class Node:
    """A single operation in the traced computation graph.

    Attributes:
        operator: Lowercased operator name (e.g. ``"aten::linear"``).
        attributes: Compile-time attributes attached to the operator.
        inputs: Input variables to this operation.
        outputs: Output variables produced by this operation.
        scope: Module scope path (e.g. ``"model/layer1/conv"``).
    """

    operator: str
    attributes: dict[str, Any]
    inputs: list[Variable]
    outputs: list[Variable]
    scope: str

    def __post_init__(self) -> None:
        self.operator = self.operator.lower()

    def __repr__(self) -> str:
        text = ", ".join(str(v) for v in self.outputs)
        text += " = " + self.operator
        if self.attributes:
            text += "[" + ", ".join(f"{k} = {v}" for k, v in self.attributes.items()) + "]"
        text += "(" + ", ".join(str(v) for v in self.inputs) + ")"
        return text


@dataclass(eq=False, repr=False)
class Graph:
    """A traced computation graph consisting of nodes and variables.

    Attributes:
        name: Qualified class name of the traced model.
        variables: All variables in the graph.
        inputs: Graph-level input variables.
        outputs: Graph-level output variables.
        nodes: Ordered list of operations.
    """

    name: str
    variables: list[Variable]
    inputs: list[Variable]
    outputs: list[Variable]
    nodes: list[Node]

    def __repr__(self) -> str:
        text = self.name + " (\n"
        text += ",\n".join(f"\t{v}" for v in self.inputs) + "\n"
        text += "):\n"
        text += "\n".join(f"\t{node}" for node in self.nodes) + "\n"
        text += "\treturn " + ", ".join(str(v) for v in self.outputs)
        return text
