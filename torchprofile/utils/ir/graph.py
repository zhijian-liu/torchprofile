__all__ = ['Graph']


class Graph:
    def __init__(self, name, variables, inputs, outputs, nodes):
        self.name = name
        self.variables = variables
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, variables):
        self._variables = variables

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

    def __repr__(self):
        text = self.name
        text += ' (' + '\n'
        text += ',\n'.join(['\t' + str(v) for v in self.inputs]) + '\n'
        text += '):' + '\n'
        text += '\n'.join(['\t' + str(x) for x in self.nodes]) + '\n'
        text += '\t' + 'return ' + ', '.join([str(v) for v in self.outputs])
        return text
