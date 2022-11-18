'''Make Poincaré maps.'''

import numpy

from . import _solution
from . import _solver
from .. import _utility


class Map:
    '''A Poincaré map.'''

    def __init__(self, func, period, t_0, t_step, **kwds):
        self.solver = _solver.Solver.create(func, **kwds)
        self.t = _utility.arange(t_0, t_0 + period, t_step)

    def build_y(self, y_0):
        '''Build storage for intermediate y values.'''
        return numpy.empty((len(self.t), *numpy.shape(y_0)))

    def solve(self, y_0, y=None, _solution_wrap=True):
        '''Get the solution y(t) over one period.'''
        return self.solver(self.t, y_0, y=y, _solution_wrap=_solution_wrap)

    def __call__(self, y_0, y=None, _solution_wrap=True):
        '''Get the solution one period later.'''
        if y is None:
            y = self.build_y(y_0)
        self.solve(y_0, y=y, _solution_wrap=False)
        y_period = y[-1]
        if _solution_wrap:
            return _solution.Solution(y_period, states=self.solver.states)
        else:
            return y_period
