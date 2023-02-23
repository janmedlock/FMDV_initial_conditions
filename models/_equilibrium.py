'''Find equilibria.'''

import numpy
import scipy.optimize

from . import _utility


def _objective(y_cur, solver, t, weights):
    '''Helper for `find`.'''
    y_new = solver.step(t, y_cur)
    diff = (y_new - y_cur) * weights
    return diff


def find(model, y_guess, t=0, weights=1, **root_kwds):
    '''Find an equilibrium `y` while keeping
    `weighted_sum(y, weights)` constant.'''
    # Ensure `y_guess` is nonnegative.
    y_guess = numpy.clip(y_guess, 0, None)
    result = scipy.optimize.root(_objective, y_guess,
                                 args=(model._solver, t, weights),
                                 **root_kwds)
    assert result.success, result
    y = result.x
    # Scale `y` so that `weighted_sum()` is the same as for
    # `y_guess`.
    y *= (_utility.weighted_sum(y_guess, weights)
          / _utility.weighted_sum(y, weights))
    return y


def eigenvalues(model, equilibrium, t=0):
    '''Get the eigenvalues of `equilibrium`.'''
    jacobian = model._solver.jacobian(t, equilibrium, equilibrium)
    evals = numpy.linalg.eigvals(jacobian)
    return _utility.sort_by_real_part(evals)
