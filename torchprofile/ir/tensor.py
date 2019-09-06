__all__ = ['Tensor']


class Tensor:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype.lower()
        self.shape = shape

    def __str__(self):
        return '%{}'.format(self.name)
