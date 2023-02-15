'''Based on our FMDV work, this is an age-structured model.'''

import numpy
import pandas

from . import _solver
from .. import model
from .. import _population
from .. import _utility


class Model(model.Base):
    '''Age-structured model.'''

    def __init__(self, age_step=0.1, age_max=50, **kwds):
        super().__init__(**kwds)
        self.age_step = age_step
        self.ages = _utility.build_t(0, age_max, self.age_step)

    def Solution(self, y, t=None):
        '''A solution.'''
        states = pandas.CategoricalIndex(self.states, self.states,
                                         ordered=True, name='state')
        ages = pandas.Index(self.ages, name='age')
        states_ages = pandas.MultiIndex.from_product((states, ages))
        if t is None:
            return pandas.Series(y, index=states_ages)
        else:
            t = pandas.Index(t, name='time')
            return pandas.DataFrame(y, index=t, columns=states_ages)

    def stable_age_density(self, *args, **kwds):
        '''Get the stable age density.'''
        (ages, v0) = _population.stable_age_density(self.birth, self.death,
                                                    *args, **kwds)
        # Interpolate the logarithm of `v0` to `self.ages`.
        assert numpy.all(v0 > 0)
        logn = numpy.interp(self.ages, ages, numpy.log(v0))
        return numpy.exp(logn)

    def build_initial_conditions(self, *args, **kwds):
        '''Build the initial conditions.'''
        # Totals over age in each immune state.
        M_bar = 0
        E_bar = 0
        I_bar = 0.01
        R_bar = 0
        S_bar = 1 - M_bar - E_bar - I_bar - R_bar
        N = self.stable_age_density(*args, **kwds)
        # X_bar * N then stacked in one big vector.
        return numpy.kron((M_bar, S_bar, E_bar, I_bar, R_bar), N)

    def solver(self):
        '''Only initialize the solver once.'''
        return _solver.Solver(self)

    def solve(self, t_span,
              y_start=None, t=None, y=None, _solution_wrap=True):
        '''Solve the ODEs.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        solver = self.solver()
        soln = solver(t_span, y_start,
                      t=t, y=y, _solution_wrap=_solution_wrap)
        _utility.assert_nonnegative(soln)
        return soln

    def __call__(self, t, y):
        raise NotImplementedError

    def jacobian(self, t, y):
        raise NotImplementedError
