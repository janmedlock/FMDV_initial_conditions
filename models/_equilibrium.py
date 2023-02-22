'''Find equilibria.'''

import numpy
import scipy.optimize

from . import _utility


def _objective(y_cur, solver, t, weights):
    '''Helper for `find`.'''
    y_new = solver.step(t, y_cur)
    return (y_new - y_cur) * weights


def find(model, y_guess, t, weights=1, **root_kwds):
    '''Find an equilibrium `y` while keeping
    `weighted_sum(y, weights)` constant.'''
    result = scipy.optimize.root(_objective, y_guess,
                                 args=(model._solver, t, weights),
                                 **root_kwds)
    assert result.success, result
    y = result.x
    # Scale `y` so that `weighted_sum()` is the same as for
    # `y_guess`.
    y *= (_utility.weighted_sum(y_guess, weights)
          / _utility.weighted_sum(y, weights))
    return model.Solution(y)


def eigenvalues(model, t, equilibrium):
    '''Get the eigenvalues of `equilibrium`.'''
    evals = numpy.linalg.eigvals(model.jacobian(t, equilibrium))
    return _utility.sort_by_real_part(evals)
