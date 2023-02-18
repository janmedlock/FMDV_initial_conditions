'''Poincaré maps.'''

import numpy

from .. import _utility


class Map:
    '''A Poincaré map.'''

    def __init__(self, model, period, t_0):
        self.model = model
        t_step = self.model.t_step
        self.t_span = (t_0, t_0 + period)
        self.t = _utility.build_t(*self.t_span, t_step)

    def build_y(self, y_0):
        '''Build storage for intermediate y values.'''
        return numpy.empty((len(self.t), *numpy.shape(y_0)))

    def solve(self, y_0, y=None, _solution_wrap=True):
        '''Get the solution y(t) over one period, not just at the end
        time.'''
        return self.model._solver.solve(self.t_span, y_0,
                                        t=self.t, y=y,
                                        _solution_wrap=_solution_wrap)

    def __call__(self, y_0, y=None, _solution_wrap=True):
        '''Get the solution one period later.'''
        if y is None:
            y = self.build_y(y_0)
        self.solve(y_0, y=y, _solution_wrap=False)
        y_period = y[-1]
        if _solution_wrap:
            return self.model.Solution(y_period)
        else:
            return y_period
