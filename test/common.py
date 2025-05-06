#!/usr/bin/python3
'''Common testing code.'''


import abc

import pytest
import pytest_dependency


def _is_exception(obj):
    return isinstance(obj, Exception)


def _raise_if_exception(obj):
    if _is_exception(obj):
        raise obj


def _with_params(*params):
    '''Format parameters for `pytest_dependency.depends()`.'''
    return f'[{"-".join(map(str, params))}]' if len(params) > 0 else ''


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
    def limit_set(self, model, solution):
        '''Limit set.'''
        try:
            assert not _is_exception(solution)
            return model.find_limit_set(self.t_end, solution.loc[self.t_end])
        except Exception as exception:
            return exception

    @pytest.fixture(scope='class')
    def exponents(self, model, limit_set):
        '''Exponents on the limit set.'''
        try:
            assert not _is_exception(limit_set)
            return model.get_exponents(limit_set)
        except Exception as exception:
            return exception

    @pytest.mark.dependency
    def test_solution(self, solution):
        '''Test solution.'''
        _raise_if_exception(solution)

    @pytest.mark.dependency
    def test_limit_set(self, request, SAT, birth_rate_constant, limit_set):
        '''Test limit set.'''
        pytest_dependency.depends(
            request,
            [f'test_solution{_with_params(SAT, birth_rate_constant)}'],
            scope='class'
        )
        _raise_if_exception(limit_set)

    @pytest.mark.dependency
    def test_exponents(self, request, SAT, birth_rate_constant, exponents):
        '''Test exponents.'''
        pytest_dependency.depends(
            request,
            [f'test_limit_set{_with_params(SAT, birth_rate_constant)}'],
            scope='class'
        )
        _raise_if_exception(exponents)
