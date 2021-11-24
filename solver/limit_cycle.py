'''Find limit cycles.'''

import numpy
import scipy.optimize

from . import solver
from . import utility


def _poincaré_map(y_0, solver_, t, y):
    '''Find the state after one period.'''
    # Solve starting from `y_0`.
    solver_(t, y_0, y=y, _solution=False)
    return y[-1]


def find(func, t_0, period, t_step, y_0_guess, **kwds):
    '''Find a limit cycle.'''
    # Instead of calling `solver.solve()`, reuse many of the
    # `solver.Solver()` parts for speed.
    solver_ = solver.Solver.create(func, **kwds)
    # Intermediate t values.
    t = utility.arange(t_0, t_0 + period, t_step)
    # Storage for intermediate y values.
    y = numpy.empty((len(t), *numpy.shape(y_0_guess)))
    # Find the `y_0` value that is a fixed point of the Poincaré map,
    # i.e. that gives `y(t_0 + period) = y_0`.
    y_0 = scipy.optimize.fixed_point(_poincaré_map,
                                     y_0_guess,
                                     args=(solver_, t, y))
    # Return the solution at all of the intermediate t values.
    return solver_(t, y_0, y=y)
