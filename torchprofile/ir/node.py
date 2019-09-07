__all__ = ['Node']


class Node:
    def __init__(self, operator, attributes, inputs, outputs, scope):
        self._operator = operator
        self._attributes = attributes
        self._inputs = inputs
        self._outputs = outputs
        self._scope = scope

    @property
    def operator(self):
        return self._operator.lower()

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
        text = ''
        if self.outputs:
            text += ', '.join([str(tensor) for tensor in self.outputs])
        text += ' = ' + self.operator
        if self.attributes:
            text += '[' + ', '.join(['{}={}'.format(k, v) for k, v in self.attributes.items()])+ ']'
        if self.inputs:
            text += '(' + ', '.join([str(tensor) for tensor in self.inputs]) + ')'
        if self.scope:
            text += ', scope={}'.format(self.scope)
        return text
        #
        # return '{} = {}({}), scope={}'.format(', '.join([str(tensor) for tensor in self.outputs]), self.operator,
        #                                       ', '.join([str(tensor) for tensor in self.inputs]), self.scope)
