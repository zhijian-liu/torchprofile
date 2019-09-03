__all__ = ['Tensor', 'Node']


class Tensor:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class Node:
    def __init__(self, operator, attrs, inputs, outputs, scope):
        self.operator = operator.lower()
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs
        self.scope = scope.lower()
