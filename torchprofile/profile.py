from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from .handlers import HANDLER_MAP
from .ir import Node
from .trace import trace


def profile_macs(
    model: nn.Module,
    args: Any = (),
    kwargs: dict[str, Any] | None = None,
    reduction: Callable | None = sum,
) -> int | dict[Node, int]:
    """Profile the number of multiply-accumulate operations (MACs) in a model.

    Uses ``torch.jit.trace`` to capture the computation graph, then counts
    MACs for each recognized operator.

    Args:
        model: The PyTorch model to profile.
        args: Positional arguments to the model. A single tensor or a tuple
            of tensors.
        kwargs: Keyword arguments (not yet supported, must be ``None``).
        reduction: Function to reduce per-operator MACs into a single value.
            Defaults to :func:`sum`. Pass ``None`` to get a per-operator
            breakdown as a ``dict[Node, int]``.

    Returns:
        Total MACs as an ``int`` when *reduction* is provided, or a ``dict``
        mapping each :class:`~torchprofile.ir.Node` to its MAC count when
        *reduction* is ``None``.
    """
    results: dict[Node, int] = {}

    graph = trace(model, args, kwargs)
    for node in graph.nodes:
        if node.operator in HANDLER_MAP:
            func = HANDLER_MAP[node.operator]
            if func is not None:
                results[node] = func(node)
        else:
            warnings.warn(f'No handlers found: "{node.operator}". Skipped.')

    if reduction is not None:
        return reduction(results.values())
    return results
