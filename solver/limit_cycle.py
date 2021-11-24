'''Find limit cycles.'''

import numpy
import scipy.optimize

from . import solver
from . import utility


def _objective(x, solver_, t, y, transform):
    '''Find the state after one period.'''
    y_0 = transform.inverse(x)
    solver_(t, y_0, y=y, _solution=False)
    return transform(y[-1])


def find(func, t_0, period, t_step, y_0_guess, **kwds):
    '''Find a limit cycle while keeping `y_0.sum()` constant.'''
    # Instead of calling `solver.solve()`, reuse many of the
    # `solver.Solver()` parts for speed.
    solver_ = solver.Solver.create(func, **kwds)
    # Intermediate t values.
    t = utility.arange(t_0, t_0 + period, t_step)
    # Storage for intermediate y values.
    y = numpy.empty((len(t), *numpy.shape(y_0_guess)))
    # Find the `y_0` value that is a fixed point of the Poincaré map,
    # i.e. that gives `y(t_0 + period) = y_0`, while keeping
    # `y_0.sum()` constant.
    transform = utility.TransformConstantSum(y_0_guess)
    x = scipy.optimize.fixed_point(_objective,
                                   transform(y_0_guess),
                                   args=(solver_, t, y, transform))
    y_0 = transform.inverse(x)
    # Return the solution at all of the intermediate t values.
    return solver_(t, y_0, y=y)
