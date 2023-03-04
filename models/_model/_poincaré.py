'''Poincaré maps.'''

import numpy

from .. import _utility


class Map:
    '''A Poincaré map.'''

    def __init__(self, model, period, t_0):
        self._solver = model._solver
        self.t_span = (t_0, t_0 + period)
        self.t = _utility.numerical.build_t(*self.t_span, self._solver.t_step)
        # Use an initial condition to determine the shape for
        # `y_temp`.
        y_start = model.build_initial_conditions()
        self.y_temp = numpy.empty((2, *numpy.shape(y_start)))

    def solve(self, y_0):
        '''Get the solution y(t) over one period, not just at the end
        time.'''
        return self._solver.solve(self.t_span, y_0, t=self.t)

    def __call__(self, y_0):
        '''Get the solution at the end of one period.'''
        return self._solver.solution_at_t_end(self.t_span, y_0,
                                              t=self.t,
                                              y_temp=self.y_temp)
