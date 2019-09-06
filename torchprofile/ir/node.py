__all__ = ['Node']


class Node:
    def __init__(self, operator, attributes, inputs, outputs, scope):
        self.operator = operator.lower()
        self.attributes = attributes
        self.inputs = inputs
        self.outputs = outputs
        self.scope = scope.lower()

    def __str__(self):
        return '{} = {}({})'.format(', '.join([str(tensor) for tensor in self.outputs]), self.operator,
                                    ', '.join([str(tensor) for tensor in self.inputs]))
