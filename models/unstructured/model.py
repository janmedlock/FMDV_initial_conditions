'''Based on our FMDV work, this is an unstructured model.'''

import numpy
import pandas

from . import _limit_cycle
from . import _solver
from .. import _equilibrium
from .. import _model
from .. import _utility


class Model(_model.AgeIndependent):
    '''Unstructured model.'''

    def __init__(self, t_step=0.001, **kwds):
        super().__init__(**kwds)
        self.t_step = t_step
        self._solver = _solver.Solver(self)

    def Solution(self, y, t=None):
        '''A solution.'''
        states = pandas.CategoricalIndex(self.states, self.states,
                                         ordered=True, name='state')
        if t is None:
            return pandas.Series(y, index=states)
        else:
            t = pandas.Index(t, name='time')
            return pandas.DataFrame(y, index=t, columns=states)

    @staticmethod
    def build_initial_conditions():
        '''Build the initial conditions.'''
        M = 0
        E = 0
        I = 0.01
        R = 0
        S = 1 - M - E - I - R
        return (M, S, E, I, R)

    def solve(self, t_span,
              y_start=None, t=None, y=None, _solution_wrap=True):
        '''Solve the ODEs.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        soln = self._solver.solve(t_span, y_start,
                                  t=t, y=y, _solution_wrap=_solution_wrap)
        _utility.assert_nonnegative(soln)
        return soln

    def find_equilibrium(self, eql_guess, t=0):
        '''Find an equilibrium of the model.'''
        eql = _equilibrium.find(self, eql_guess, t)
        _utility.assert_nonnegative(eql)
        return eql

    def get_eigenvalues(self, eql):
        '''Get the eigenvalues of the Jacobian.'''
        return _equilibrium.eigenvalues(self, 0, eql)

    def find_limit_cycle(self, period_0, t_0, lcy_0_guess):
        '''Find a limit cycle of the model.'''
        lcy = _limit_cycle.find_subharmonic(self, period_0, t_0, lcy_0_guess)
        _utility.assert_nonnegative(lcy)
        return lcy

    def get_characteristic_multipliers(self, lcy):
        '''Get the characteristic multipliers.'''
        return _limit_cycle.characteristic_multipliers(self, lcy)

    def get_characteristic_exponents(self, lcy):
        '''Get the characteristic exponents.'''
        return _limit_cycle.characteristic_exponents(self, lcy)

    def jacobian(self, t, y):
        '''The Jacobian of the model.'''
        (M, S, E, I, R) = y
        birth_rate_t = self.birth.rate(t)
        dM = (birth_rate_t * self._states_have_antibodies
              + numpy.array((- 1 / self.waning.mean - self.death_rate_mean,
                             0,
                             0,
                             0,
                             0)))
        dS = (birth_rate_t * ~self._states_have_antibodies
              + numpy.array((1 / self.waning.mean,
                             (- self.transmission.rate * I
                              - self.death_rate_mean),
                             0,
                             - self.transmission.rate * S,
                             0)))
        dE = numpy.array((0,
                          self.transmission.rate * I,
                          - 1 / self.progression.mean - self.death_rate_mean,
                          self.transmission.rate * S,
                          0))
        dI = numpy.array((0,
                          0,
                          1 / self.progression.mean,
                          - 1 / self.recovery.mean - self.death_rate_mean,
                          0))
        dR = numpy.array((0,
                          0,
                          0,
                          1 / self.recovery.mean,
                          - self.death_rate_mean))
        return numpy.vstack((dM, dS, dE, dI, dR))
