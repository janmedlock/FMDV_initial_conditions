'''Based on our FMDV work, this is an age-structured model.'''

import numpy

from . import _solver
from .. import model
from .. import _population
from .. import _utility


class Model(model.Base):
    '''Age-structured model.'''

    def __init__(self, age_step=0.1, age_max=50, **kwds):
        super().__init__(**kwds)
        self.age_step = age_step
        self.age_max = age_max
        self.ages = _utility.build_t(0, self.age_max, self.age_step)

    @staticmethod
    def stack(arrs):
        return numpy.concatenate(arrs)

    def split(self, y):
        return numpy.split(y, len(self.states))

    def __call__(self, t, y):
        raise NotImplementedError

    def jacobian(self, t, y):
        raise NotImplementedError

    def stable_age_density(self):
        '''Get the stable age density.'''
        (_, N) = _population.stable_age_density(self.birth, self.death,
                                                age_step=self.age_step,
                                                age_max=self.age_max)
        return N

    def build_initial_conditions(self):
        '''Build the initial conditions.'''
        # Means over age in each immune state.
        M_bar = 0
        E_bar = 0
        I_bar = 0.01
        R_bar = 0
        S_bar = 1 - M_bar - E_bar - I_bar - R_bar
        return numpy.kron((M_bar, S_bar, E_bar, I_bar, R_bar),
                          self.stable_age_density())

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
