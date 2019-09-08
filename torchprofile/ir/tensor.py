__all__ = ['Tensor']


class Tensor:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.value = None
        self.dtype = dtype
        self.shape = shape

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype.lower()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def size(self):
        return self._shape

    def __repr__(self):
        text = '%' + self.name + ' : ' + self.dtype
        if self.shape:
            text += '(' + ', '.join([str(size) for size in self.shape]) + ')'
        if self.value:
            text += ' = ' + str(self.value)
        return text
