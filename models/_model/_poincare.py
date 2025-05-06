'''Poincaré maps.'''

import numpy

from .. import _utility
from .._utility import _transform


class Map:
    '''A Poincaré map.'''

    def __init__(self, model, period, t_0):
        self.solver = model.solver
        self.t_span = (t_0, t_0 + period)
        self.t = _utility.numerical.build_t(*self.t_span, self.solver.t_step)
        # Use an initial condition to determine the shape for
        # `y_temp`.
        y_start = model.build_initial_conditions()
        self.y_temp = numpy.empty((2, *numpy.shape(y_start)))

    @property
    def sparse(self):
        '''Whether the solver uses sparse arrays and sparse linear
        algebra.'''
        return self.solver.sparse

    def solve(self, y_0, display=False):
        '''Get the solution y(t) over one period, not just at the end
        time.'''
        return self.solver.solve(self.t_span, y_0,
                                 t=self.t, display=display)

    def __call__(self, y_0, display=False):
        '''Get the solution at the end of one period.'''
        return self.solver.solution_at_t_end(self.t_span, y_0,
                                             t=self.t,
                                             y_temp=self.y_temp,
                                             display=display)

    def _root_objective(self, x_0_cur, weights, transform, display):
        '''Helper for `.find_fixed_point(..., solver='root', ...)`.'''
        y_0_cur = transform.inverse(x_0_cur)
        y_0_new = self(y_0_cur, display=display)
        diff = (y_0_new - y_0_cur) * weights
        return diff

    def _fixed_point_objective(self, x_0_cur, transform, display):
        '''Helper for `.find_fixed_point(..., solver='root', ...)`.'''
        y_0_cur = transform.inverse(x_0_cur)
        y_0_new = self(y_0_cur, display=display)
        x_0_new = transform(y_0_new)
        return x_0_new

    def find_fixed_point(self, y_0_guess,
                         solver='root', weights=1, display=False,
                         **kwds):
        '''Find a fixed point `y_0` of the Poincaré map, i.e. that gives
        `y(t_0 + period) = y_0`.'''
        # TODO: a=0?
        transform = _transform.Logarithm(a=1e-6,
                                         weights=weights)
        x_0_guess = transform(y_0_guess)
        if solver == 'root':
            x_0 = _utility.optimize.root(self._root_objective, x_0_guess,
                                         args=(weights, transform, display),
                                         sparse=self.sparse,
                                         display=display,
                                         **kwds)
        elif solver == 'fixed_point':
            x_0 = _utility.optimize.fixed_point(self._fixed_point_objective,
                                                x_0_guess,
                                                args=(transform, display),
                                                **kwds)
        else:
            raise ValueError(f'Unknown {solver=}!')
        y_0 = transform.inverse(x_0)
        # Scale `y_0` so that `weighted_sum()` is the same as for
        # `y_0_guess`.
        # TODO: Is this wrong? Does it scale correctly for infection?
        y_0 *= (_utility.numerical.weighted_sum(y_0_guess, weights)
                / _utility.numerical.weighted_sum(y_0, weights))
        return y_0
