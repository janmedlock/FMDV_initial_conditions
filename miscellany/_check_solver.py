'''Utilities for checking the model solver matrices.'''

import abc

from context import models
from models import _utility


class Base:
    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def beta(self): pass

    @abc.abstractmethod
    def H(self, q): pass

    @abc.abstractmethod
    def F(self, q): pass

    @abc.abstractmethod
    def T(self, q): pass

    @abc.abstractmethod
    def B(self): pass

    def check_matrices(self):
        solver = self.model._solver
        names = ('beta', 'H', 'F', 'T', 'B')
        for name in names:
            matrix = getattr(solver, name)
            checker = getattr(self, name)
            if isinstance(matrix, dict):
                for q in ('new', 'cur'):
                    assert _utility.sparse.equals(matrix[q], checker(q))
            else:
                assert _utility.sparse.equals(matrix, checker())
