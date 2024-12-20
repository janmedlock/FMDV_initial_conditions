'''Mixins for age-structured model and their solvers.'''

import functools

import numpy

from .. import _utility


class ModelMixin:
    '''Mixin for age-structured models.'''

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


class SolverMixin:
    '''Mixin for solvers of age-structured models.'''

    sparse = True

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

    @functools.cached_property
    def _zeros_a(self):
        return _utility.sparse.Array((1, len(self.a)))

    @functools.cached_property
    def _iota_a(self):
        '''The block for integrating over age.'''
        return self._integration_vector(len(self.a), self.a_step)

    @functools.cached_property
    def _zeta_a(self):
        '''The block for influx for age.'''
        return self._influx_vector(len(self.a), self.a_step)

    @functools.cached_property
    def _tau_a(self):
        '''The maternity integration vector over age.'''
        return self._integration_against_vector(
            len(self.a), self.a_step,
            self.parameters.birth.maternity(self.a)
        )

    @functools.cached_property
    def _Zeros_a_a(self):  # pylint: disable=invalid-name
        '''Zero matrix used in constructing the other matrices.'''
        return self._zeros_a.T @ self._zeros_a

    @functools.cached_property
    def _I_a(self):  # pylint: disable=invalid-name
        '''The identity matrix block for age.'''
        return _utility.sparse.identity(len(self.a))

    @functools.cached_property
    def _L_a(self):  # pylint: disable=invalid-name
        '''The lag matrix block for age.'''
        return self._lag_matrix(len(self.a))

    def _H_a(self, q):  # pylint: disable=invalid-name
        '''The age diagonal block of H(q).'''
        if q == 'new':
            H_a = self._I_a  # pylint: disable=invalid-name
        elif q == 'cur':
            H_a = self._L_a  # pylint: disable=invalid-name
        else:
            raise ValueError(f'{q=}!')
        return H_a

    def _F_a(self, q, pi):  # pylint: disable=invalid-name
        '''An age block of F(q).'''
        if q not in self._q_vals:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(pi):
            pi *= numpy.ones(len(self.a))
        F_a = _utility.sparse.diags(pi)  # pylint: disable=invalid-name
        if q == 'cur':
            F_a = self._L_a @ F_a  # pylint: disable=invalid-name
        return F_a

    @functools.cached_property
    def _B_a(self):  # pylint: disable=invalid-name
        '''The age block of B.'''
        return self._zeta_a @ self._tau_a

    def _get_rate_a(self, which):
        '''Get the age-dependent rate `which` and make finite any
        infinite entries.'''
        waiting_time = getattr(self.model.parameters, which)
        rate = waiting_time.rate(self.a)
        return _utility.numerical.rate_make_finite(rate)
