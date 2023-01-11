'''Find equilibria.'''

import numpy
import scipy.optimize

from . import _solution
from .. import _utility


def _objective(x, model, t, transform):
    y = transform.inverse(x)
    dy = model(t, y)
    return transform(dy)


def find(model, t, y_guess):
    '''Find an equilibrium while keeping `y_guess.sum()` constant.'''
    transform = _utility.TransformConstantSum(y_guess)
    result = scipy.optimize.root(_objective,
                                 transform(y_guess),
                                 args=(model, t, transform))
    assert result.success, f'{result}'
    y = transform.inverse(result.x)
    return _solution.Solution(y, states=model.states)


def eigenvalues(model, t, equilibrium):
    '''Get the eigenvalues of `equilibrium`.'''
    jacobian = _utility.jacobian(model)
    evals = numpy.linalg.eigvals(jacobian(t, equilibrium))
    return _utility.sort_by_real_part(evals)
