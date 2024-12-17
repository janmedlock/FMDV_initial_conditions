'''Utilities for testing the model solver matrices.'''

import abc

from context import models
import models._utility


SparseArray = models._utility.sparse.Array


class Tester:
    '''Base tester for solver matrices.'''

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def beta(self):
        pass

    @abc.abstractmethod
    def A(self, q):
        pass

    @abc.abstractmethod
    def B(self):
        pass

    @abc.abstractmethod
    def T(self, q):
        pass

    def test(self):
        solver = self.model._solver
        names = ('beta', 'A', 'B', 'T')
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
