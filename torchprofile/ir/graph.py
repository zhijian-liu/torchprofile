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
        return self.name + ' (\n' + \
               ',\n'.join(['  ' + str(var) for var in self.inputs]) + '\n):\n' + \
               '\n'.join(['  ' + str(node) for node in self.nodes]) + '\n  return ' + \
               ', '.join([str(var) for var in self.outputs])
