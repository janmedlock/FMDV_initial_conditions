#!/usr/bin/python3
'''Common testing code.'''


import abc
import functools

import pytest


class TestModel(metaclass=abc.ABCMeta):
    '''Base class for testing a model.'''

    t_start = 0
    t_end = 20

    @property
    @abc.abstractmethod
    def Model(self):
        '''The model class to test.'''
        raise NotImplementedError

    @pytest.fixture(params=[1, 2, 3], scope='class')
    def SAT(self, request):
        '''`SAT`.'''
        return request.param

    @pytest.fixture(params=[True, False], scope='class')
    def birth_rate_constant(self, request):
        '''Whether `birth_rate` is constant.'''
        return request.param

    @pytest.fixture(scope='class')
    def model(self, SAT, birth_rate_constant):
        '''Model instance.'''
        parameters = {
            'SAT': SAT,
        }
        if birth_rate_constant:
            parameters['birth_variation'] = 0
        return self.Model(**parameters)

    @pytest.fixture(scope='class')
    def solution(self, model):
        '''Solution.'''
        try:
            return model.solve((self.t_start, self.t_end))
        except Exception as exception:
            return exception

    @pytest.fixture(scope='class')
    def limit_set(self, birth_rate_constant, model, solution):
        '''Limit set.'''
        if isinstance(solution, Exception):
            return None
        if birth_rate_constant:
            fcn = model.find_equilibrium
        else:
            period = model.parameters.period
            fcn = functools.partial(model.find_limit_cycle,
                                    period, self.t_end % period)
        guess = solution.loc[self.t_end]
        try:
            return fcn(guess)
        except Exception as exception:
            return exception

    @pytest.fixture(scope='class')
    def exponents(self, birth_rate_constant, model, limit_set):
        '''Exponents on the limit set.'''
        if isinstance(limit_set, Exception) or (limit_set is None):
            return None
        if birth_rate_constant:
            fcn = model.get_eigenvalues
        else:
            fcn = model.get_characteristic_exponents
        try:
            return fcn(limit_set)
        except Exception as exception:
            return exception

    def _test_arg(self, arg):
        if arg is None:
            pytest.skip()
        else:
            assert not isinstance(arg, Exception)

    def test_solution(self, solution):
        '''Test solution.'''
        self._test_arg(solution)

    def test_limit_set(self, limit_set):
        '''Test limit set.'''
        self._test_arg(limit_set)

    def test_exponents(self, exponents):
        '''Test exponents.'''
        self._test_arg(exponents)
