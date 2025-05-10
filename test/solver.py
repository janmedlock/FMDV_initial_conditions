'''Test solver matrices.'''

import abc

import pytest

from context import models
from models._utility import numerical
from models._utility import sparse


SparseArray = sparse.Array


def Zeros(shape):
    '''Zeros.'''
    return SparseArray(shape)


class TestSolver(metaclass=abc.ABCMeta):
    '''Base class for testing solver matrices.'''

    @property
    @abc.abstractmethod
    def Model(self):
        '''The model class to test.'''
        raise NotImplementedError

    @pytest.fixture(scope='class')
    def model(self):
        '''The model instance.'''
        return self.Model()

    @pytest.fixture(params=['new', 'cur'])
    def q(self, request):
        '''`q`.'''
        return request.param

    @abc.abstractmethod
    def H(self, model, q):
        pass

    @abc.abstractmethod
    def F(self, model, q):
        pass

    @abc.abstractmethod
    def B(self, model):
        pass

    def test_B(self, model):
        assert sparse.equals(model.solver.B, self.B(model))

    @abc.abstractmethod
    def beta(self, model):
        pass

    def test_beta(self, model):
        assert sparse.equals(model.solver.beta, self.beta(model))

    @abc.abstractmethod
    def T(self, model, q):
        pass

    def test_T(self, model, q):
        assert sparse.equals(model.solver.T[q], self.T(model, q))

    def A(self, model, q):
        if q == 'new':
            A_ = self.H(model, q) - model.t_step / 2 * self.F(model, q)
        elif q == 'cur':
            A_ = self.H(model, q) + model.t_step / 2 * self.F(model, q)
        else:
            raise ValueError(f'{q=}')
        return A_

    def test_A(self, model, q):
        assert sparse.equals(model.solver.A[q], self.A(model, q))
