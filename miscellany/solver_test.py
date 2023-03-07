'''Utilities for testing the model solver matrices.'''

import abc

import scipy.sparse

from context import models
import models._utility


sparse_array = models._utility.sparse.array


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

    def test(self):
        solver = self.model._solver
        names = ('beta', 'H', 'F', 'T', 'B')
        for name in names:
            matrix = getattr(solver, name)
            test_fcn = getattr(self, name)
            if isinstance(matrix, dict):
                for q in ('new', 'cur'):
                    assert models._utility.sparse.equals(matrix[q],
                                                         test_fcn(q))
            else:
                assert models._utility.sparse.equals(matrix,
                                                     test_fcn())
