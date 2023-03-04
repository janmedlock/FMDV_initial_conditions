'''Age-structured population model.'''

import numpy

from . import _solver
from .. import _age
from ... import _utility


class Model:
    '''Solver for the monodromy matrix of a linear age-structured
    model for the population size with age-dependent death rate,
    age-dependent maternity, and periodic time-dependent birth rate.'''

    # The default time step `t_step`.
    # TODO: HOW WAS IT CHOSEN?
    _t_step_default = 1e-1

    def __init__(self, birth, death,
                 t_step=_t_step_default,
                 a_max=_age.max_default):
        self.birth = birth
        self.death = death
        self.t_step = t_step
        self.period = self.birth.period
        if self.period == 0:
            self.period = self.t_step
        self.t = _utility.numerical.build_t(0, self.period, self.t_step)
        assert numpy.isclose(self.t[-1], self.period)
        self.a_step = _solver.Solver._get_a_step(self.t_step)
        self.a = _utility.numerical.build_t(0, a_max, self.a_step)
        _age.check_max(self)
        self._solver = _solver.Solver(self)

    def birth_scaling_for_zero_population_growth(self, **kwds):
        '''Find the birth scaling that gives zero population growth rate.'''
        return self._solver.birth_scaling_for_zero_population_growth(**kwds)

    def integral_over_a(self, arr, *args, **kwds):
        '''Integrate `arr` over age. `args` and `kwds` are passed on
         to `.sum()`.'''
        return self._solver.integral_over_a(arr, *args, **kwds)

    def stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        return self._solver.stable_age_density(**kwds)
