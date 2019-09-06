__all__ = ['Node']


class Node:
    def __init__(self, operator, attributes, inputs, outputs, scope):
        self._operator = operator.lower()
        self._attributes = attributes
        self._inputs = inputs
        self._outputs = outputs
        self._scope = scope.lower()

    @property
    def operator(self):
        return self._operator

    @property
    def attributes(self):
        return self._attributes

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def scope(self):
        return self._scope

    def __str__(self):
        return '{} = {}({})'.format(', '.join([str(tensor) for tensor in self.outputs]), self.operator,
                                    ', '.join([str(tensor) for tensor in self.inputs]))
