'''Age-structured population model.'''

from . import solver
from .. import _base
from ... import _model


class Model(_base.ModelMixin, _model.model.Population):
    '''The linear age-structured model for the population size with
    age-dependent death rate, age-dependent maternity, and periodic
    time-dependent birth rate.'''

    _Solver = solver.Solver

    # The default time step `t_step`.
    # `t_step` = 1e-2 gives a relative error in `.stable_age_density()`
    # of less than 1e-3. (See `../../miscellany/population_step_size.py`.)
    _t_step_default = 1e-2

    def __init__(self,
                 t_step=_t_step_default,
                 parameters=None,
                 **kwds):
        assert parameters is not None
        self.parameters = parameters
        super().__init__(t_step, parameters, **kwds)

    def integral_over_a(self, arr, *args, **kwds):
        '''Integrate `arr` over age. `args` and `kwds` are passed on
         to `.sum()`.'''
        return self.solver.integral_over_a(arr, *args, **kwds)

    def birth_scaling_for_zero_population_growth(self, **kwds):
        '''Find the birth scaling that gives zero population growth rate.'''
        return self.solver.birth_scaling_for_zero_population_growth(**kwds)

    def stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        return self.solver.stable_age_density(**kwds)
