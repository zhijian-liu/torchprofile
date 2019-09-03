__all__ = ['Tensor', 'Node']


class Tensor:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype.lower()
        self.shape = shape

    def __str__(self):
        return '%{}'.format(self.name)


class Node:
    def __init__(self, operator, attrs, inputs, outputs, scope):
        self.operator = operator.lower()
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs
        self.scope = scope.lower()

    def __str__(self):
        return '{} = {}({})'.format(', '.join([str(tensor) for tensor in self.outputs]), self.operator,
                                    ', '.join([str(tensor) for tensor in self.inputs]))
