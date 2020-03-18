__all__ = ['Node']


class Node:
    def __init__(self, operator, attributes, inputs, outputs, scope):
        self.operator = operator
        self.attributes = attributes
        self.inputs = inputs
        self.outputs = outputs
        self.scope = scope

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        self._operator = operator.lower()

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = attributes

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, scope):
        self._scope = scope

    def __repr__(self):
        text = ', '.join([str(v) for v in self.outputs])
        text += ' = ' + self.operator
        if self.attributes:
            text += '[' + ', '.join(
                [str(k) + ' = ' + str(v)
                 for k, v in self.attributes.items()]) + ']'
        text += '(' + ', '.join([str(v) for v in self.inputs]) + ')'
        return text
