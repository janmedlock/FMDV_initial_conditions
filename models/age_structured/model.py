'''Based on our FMDV work, this is an age-structured model.'''

import numpy
import pandas

from . import _solver
from .. import model
from .. import _equilibrium
from .. import _population
from .. import _utility


class Model(model.Base):
    '''Age-structured model.'''

    def __init__(self, a_step=0.001, a_max=25, **kwds):
        super().__init__(**kwds)
        self.a_step = a_step
        self.a = _utility.build_t(0, a_max, self.a_step)
        self._solver = _solver.Solver(self)

    def Solution(self, y, t=None):
        '''A solution.'''
        states = pandas.CategoricalIndex(self.states, self.states,
                                         ordered=True, name='state')
        ages = pandas.Index(self.a, name='age')
        states_ages = pandas.MultiIndex.from_product((states, ages))
        if t is None:
            return pandas.Series(y, index=states_ages)
        else:
            t = pandas.Index(t, name='time')
            return pandas.DataFrame(y, index=t, columns=states_ages)

    def stable_age_density(self, *args, **kwds):
        '''Get the stable age density.'''
        (a, v0) = _population.stable_age_density(self.birth, self.death,
                                                 *args, **kwds)
        # Interpolate the logarithm of `v0` to `self.a`.
        assert numpy.all(v0 > 0)
        logn = numpy.interp(self.a, a, numpy.log(v0))
        n = numpy.exp(logn)
        # Normalize to integrate to 1.
        n /= n.sum() * self.a_step
        return n

    def initial_conditions_from_unstructured(self, Y, *args, **kwds):
        '''Build initial conditions from the unstructured `Y`.'''
        n = self.stable_age_density(*args, **kwds)
        # [X * n for X in Y] stacked in one big vector.
        return numpy.kron(Y, n)

    def build_initial_conditions(self, *args, **kwds):
        '''Build the initial conditions.'''
        # Totals over age in each immune state.
        M = 0
        E = 0
        I = 0.01
        R = 0
        S = 1 - M - E - I - R
        Y = (M, S, E, I, R)
        return self.initial_conditions_from_unstructured(Y, *args, **kwds)

    def solve(self, t_span,
              y_start=None, t=None, y=None, _solution_wrap=True):
        '''Solve the ODEs.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        soln = self._solver.solve(t_span, y_start,
                                  t=t, y=y, _solution_wrap=_solution_wrap)
        _utility.assert_nonnegative(soln)
        return soln

    def find_equilibrium(self, eql_guess, t=0, **kwds):
        '''Find an equilibrium of the model.'''
        if not 'method' in kwds:
            kwds['method'] = 'krylov'
        eql = _equilibrium.find(self, eql_guess, t, **kwds)
        _utility.assert_nonnegative(eql)
        return eql
