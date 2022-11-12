'''Find equilibria.'''

import numpy
import scipy.optimize

from . import _solution
from .. import _utility


def _objective(x, func, t, transform):
    y = transform.inverse(x)
    dy = func(t, y)
    return transform(dy)


def find(func, t, y_guess, states=None):
    '''Find an equilibrium while keeping `y_guess.sum()` constant.'''
    transform = _utility.TransformConstantSum(y_guess)
    result = scipy.optimize.root(_objective,
                                 transform(y_guess),
                                 args=(func, t, transform))
    assert result.success, f'{result}'
    y = transform.inverse(result.x)
    return _solution.Solution(y, states=states)


def eigenvalues(func, t, equilibrium):
    '''Get the eigenvalues of `equilibrium`.'''
    jacobian = _utility.jacobian(func)
    evals = numpy.linalg.eigvals(jacobian(t, equilibrium))
    return _utility.sort_by_real_part(evals)
