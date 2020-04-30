import operator
from functools import reduce

__all__ = ['prod']


def prod(iterable):
    return reduce(operator.mul, iterable, 1)
