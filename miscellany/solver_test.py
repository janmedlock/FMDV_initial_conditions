'''Utilities for testing the model solver matrices.'''

import abc

from context import models
from models import _utility


SparseArray = _utility.sparse.Array


def Zeros(shape):
    '''Zeros.'''
    return SparseArray(shape)


class Tester:
    '''Base tester for solver matrices.'''

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def H(self, q):
        pass

    @abc.abstractmethod
    def F(self, q):
        pass

    @abc.abstractmethod
    def B(self):
        pass

    @abc.abstractmethod
    def beta(self):
        pass

    @abc.abstractmethod
    def T(self, q):
        pass

    def A(self, q):
        if q == 'new':
            A_ = self.H(q) - self.model.t_step / 2 * self.F(q)
        elif q == 'cur':
            A_ = self.H(q) + self.model.t_step / 2 * self.F(q)
        else:
            raise ValueError(f'{q=}')
        return A_

    def test(self):
        solver = self.model.solver
        names = ('A', 'B', 'beta', 'T')
        for name in names:
            matrix = getattr(solver, name)
            test_fcn = getattr(self, name)
            if isinstance(matrix, dict):
                for q in ('new', 'cur'):
                    assert _utility.sparse.equals(matrix[q],
                                                  test_fcn(q))
            else:
                assert _utility.sparse.equals(matrix,
                                              test_fcn())
