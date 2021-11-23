'''Find equilibria.'''

import scipy.optimize

from . import solution


def _objective(y, func, t):
    return func(t, y)


def find(func, t, y_guess, states=None):
    '''Find an equilibrium.'''
    result = scipy.optimize.root(_objective,
                                 y_guess,
                                 args=(func, t))
    assert result.success, f'{result}'
    y = result.x
    return solution.State(y, states=states)
