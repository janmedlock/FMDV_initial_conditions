'''Find equilibria.'''

import scipy.optimize

from . import solution
from . import utility


def _objective(x, func, t, transform):
    y = transform.inverse(x)
    dy = func(t, y)
    return transform(dy)


def find(func, t, y_guess, states=None):
    '''Find an equilibrium while keeping `y_guess.sum()` constant.'''
    transform = utility.TransformConstantSum(y_guess)
    result = scipy.optimize.root(_objective,
                                 transform(y_guess),
                                 args=(func, t, transform))
    assert result.success, f'{result}'
    y = transform.inverse(result.x)
    return solution.Solution(y, states=states)
