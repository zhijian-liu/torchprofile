__all__ = ['Tensor']


class Tensor:
    def __init__(self, name, dtype, shape):
        self._name = name
        self._dtype = dtype.lower()
        self._shape = shape

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def size(self):
        return self._shape

    def __str__(self):
        return '%{}'.format(self.name)
