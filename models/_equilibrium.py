'''Find equilibria.'''

import numpy
import scipy.optimize

from . import _utility


def _objective(x, solver, transform, t):
    y_cur = transform.inverse(x)
    y_new = solver.step(t, y_cur)
    return transform(y_new) - x


def find(model, y_guess, t, weights=1, **root_kwds):
    '''Find an equilibrium.'''
    # Find an equilibirium `y` while keeping `y.dot(weights)` constant.
    transform = _utility.transform.ConstantSumLogarithm.from_y(y_guess,
                                                               weights=weights)
    result = scipy.optimize.root(_objective, transform(y_guess),
                                 args=(model._solver, transform, t),
                                 **root_kwds)
    assert result.success, result
    y = transform.inverse(result.x)
    return model.Solution(y)


def eigenvalues(model, t, equilibrium):
    '''Get the eigenvalues of `equilibrium`.'''
    evals = numpy.linalg.eigvals(model.jacobian(t, equilibrium))
    return _utility.sort_by_real_part(evals)
