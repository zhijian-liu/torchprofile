import warnings

from .handlers import handlers
from .utils.trace import trace

__all__ = ['profile_macs']


def profile_macs(model, *args, reduction=sum, **kwargs):
    graph = trace(model, *args, **kwargs)

    results = dict()
    for node in graph.nodes:
        for operator, func in handlers:
            if node.operator == operator or (isinstance(operator, (list, tuple)) and node.operator in operator):
                results[node] = func(node)
                break

        if node not in results:
            warnings.warn('missing handler for {}'.format(node.operator), UserWarning)

    if reduction is not None:
        return reduction(results.values())
    else:
        return results
