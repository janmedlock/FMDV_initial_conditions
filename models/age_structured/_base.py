'''Base for age-structured models.'''

import abc
import functools

from .. import _model, _utility


class Model(_model.model.Population, metaclass=abc.ABCMeta):
    '''Base for age-structured models.'''

    @property
    def a_max(self):
        '''The maximum age.'''
        return self._solver.a_max

    @property
    def a_step(self):
        '''The step size in age.'''
        return self._solver.a_step

    @property
    def a(self):
        '''The solution ages.'''
        return self._solver.a


class Solver(_model.solver.Population, metaclass=abc.ABCMeta):
    '''Base for age-structured solvers.'''

    @property
    def a_max(self):
        '''The maximum age.'''
        return self.parameters.age_max

    @property
    def a_step(self):
        '''The step size in age.'''
        return self.t_step

    @functools.cached_property
    def a(self):
        '''The solution ages.'''
        return _utility.numerical.build_t(0, self.a_max, self.a_step)

    @property
    def _zeta(self):
        '''The influx vector.'''
        return self._influx_vector(len(self.a), self.a_step)

    @functools.cached_property
    def _I_a(self):  # pylint: disable=invalid-name
        '''The identity matrix block.'''
        return _utility.sparse.identity(len(self.a))

    @functools.cached_property
    def _L_a(self):  # pylint: disable=invalid-name
        '''The lag matrix block.'''
        return self._lag_matrix(len(self.a))

    def _H_a(self, q):  # pylint: disable=invalid-name
        '''The diagonal block of H(q).'''
        if q == 'new':
            H_a = self._I_a  # pylint: disable=invalid-name
        elif q == 'cur':
            H_a = self._L_a  # pylint: disable=invalid-name
        else:
            raise ValueError(f'{q=}!')
        return H_a

    @property
    def _tau(self):
        '''The maternity integration vector.'''
        return self._integration_against_vector(
            len(self.a), self.a_step,
            self.parameters.birth.maternity(self.a)
        )

    @property
    def _B_a(self):  # pylint: disable=invalid-name
        '''The block of B.'''
        return self._zeta @ self._tau
