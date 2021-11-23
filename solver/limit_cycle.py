'''Find limit cycles.'''

import numpy
import scipy.optimize

from . import solver
from . import utility


def _objective(y_0, solver_, t, y):
    '''Solve the ODEs and return the difference between the initial state
    and the state one period later.'''
    solver_.solve(t, y_0, y=y, _solution=False)
    return y[-1] - y_0


def find(func, t_0, period, t_step, y_0_guess, **kwds):
    '''Find a limit cycle.'''
    solver_ = solver.Solver.create(func, **kwds)
    t = utility.arange(t_0, t_0 + period, t_step)
    y = numpy.empty((len(t), *numpy.shape(y_0_guess)))
    result = scipy.optimize.root(_objective,
                                 y_0_guess,
                                 args=(solver_, t, y))
    assert result.success, f'{result}'
    y_0 = result.x
    return solver_.solve(t, y_0, y=y)
