'''Make Poincaré maps.'''

from . import solution
from . import solver
from .. import utility


class Map:
    '''A Poincaré map.'''

    def __init__(self, func, t_0, period, t_step, **kwds):
        self.solver = solver.Solver.create(func, **kwds)
        self.t = utility.arange(t_0, t_0 + period, t_step)

    def __call__(self, y_0, _solution=True, **kwds):
        '''Get the solution one period later.'''
        y = self.solver(self.t, y_0, _solution=False, **kwds)
        y_period = y[-1]
        if _solution:
            return solution.Solution(y_period, states=self.solver.states)
        else:
            return y_period
