'''Age-structured population model.'''

import functools

from . import _integral, _solver
from .. import _age


class Model:
    '''Solver for the monodromy matrix of a linear age-structured
    model for the population size with age-dependent death rate,
    age-dependent maternity, and periodic time-dependent birth rate.'''

    # The default time step `t_step`.
    # `t_step` = 1e-2 gives a relative error in `.stable_age_density()`
    # of less than 1e-3. (See `../../miscellany/population_step_size.py`.)
    _t_step_default = 1e-2

    def __init__(self, birth, death,
                 t_step=_t_step_default,
                 a_max=_age.max_default):
        self.birth = birth
        self.death = death
        assert t_step > 0
        self.t_step = t_step
        assert a_max > 0
        _age.check_max(self, a_max)
        self.a_max = a_max
        self.a_step = _solver.Solver._get_a_step(self.t_step)
        assert self.a_step > 0

    @functools.cached_property
    def _solver(self):
        '''`._solver` is built on first use and then reused.'''
        _solver_ = _solver.Solver(self)
        return _solver_

    def integral_over_a(self, arr, *args, **kwds):
        '''Integrate `arr` over age. `args` and `kwds` are passed on
         to `.sum()`.'''
        return _integral.over_a(arr, self.a_step, *args, **kwds)

    # TODO: Cache this method.
    def birth_scaling_for_zero_population_growth(self, **kwds):
        '''Find the birth scaling that gives zero population growth rate.'''
        return self._solver.birth_scaling_for_zero_population_growth(**kwds)

    # TODO: Cache this method.
    def stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        return self._solver.stable_age_density(**kwds)
