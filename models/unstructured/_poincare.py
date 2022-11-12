'''Make Poincaré maps.'''

from . import _solution
from . import _solver
from .. import _utility


class Map:
    '''A Poincaré map.'''

    def __init__(self, func, t_0, period, t_step, **kwds):
        self.solver = _solver.Solver.create(func, **kwds)
        self.t = _utility.arange(t_0, t_0 + period, t_step)

    def __call__(self, y_0, _solution_wrap=True, **kwds):
        '''Get the solution one period later.'''
        y = self.solver(self.t, y_0, _solution_wrap=False, **kwds)
        y_period = y[-1]
        if _solution_wrap:
            return _solution.Solution(y_period, states=self.solver.states)
        else:
            return y_period
