'''Find equilibria.'''

import numpy
import scipy.optimize

from . import _utility


def _objective(x, solver, t, transform, weights):
    '''Helper for `find`.'''
    y_cur = transform.inverse(x)
    y_new = solver.step(t, y_cur)
    diff = (y_new - y_cur) * weights
    return diff[:-1]


def find(model, y_guess, t, weights=1, **root_kwds):
    '''Find an equilibrium `y` while keeping
    `weighted_sum(y, weights)` constant.'''
    # Ensure `y_guess` is nonnegative.
    y_guess = numpy.clip(y_guess, 0, None)
    # Transform `y` to simplex coordinates and clip `x_guess` away
    # from infinity.
    transform = _utility.transform.Simplex(weights=weights)
    x_guess = transform(y_guess).clip(-10, 10)
    result = scipy.optimize.root(_objective, x_guess,
                                 args=(model._solver, t, transform, weights),
                                 **root_kwds)
    assert result.success, result
    y = transform.inverse(result.x)
    # Scale `y` so that `weighted_sum()` is the same as for
    # `y_guess`.
    y *= (_utility.weighted_sum(y_guess, weights)
          / _utility.weighted_sum(y, weights))
    return model.Solution(y)


def eigenvalues(model, t, equilibrium):
    '''Get the eigenvalues of `equilibrium`.'''
    evals = numpy.linalg.eigvals(model.jacobian(t, equilibrium))
    return _utility.sort_by_real_part(evals)
