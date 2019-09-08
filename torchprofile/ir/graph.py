__all__ = ['Graph']


class Graph:
    def __init__(self, variables, inputs, outputs, nodes):
        self.variables = variables
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes

    @property
    def variables(self):
        return self._tensors

    @variables.setter
    def variables(self, tensors):
        self._tensors = tensors

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
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    def __str__(self):
        text = '\n'.join([str(node) for node in self.nodes])
        if self.outputs:
            text += '\n' + 'return ' + '(' + ', '.join(['%' + tensor.name for tensor in self.outputs]) + ')'
        return text
