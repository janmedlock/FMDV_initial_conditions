'''Find equilibria.'''

import numpy

from .._utility import linalg, numerical, optimize


def _solve(model, y_0, t, t_solve):
    '''Solve from `t` to `t + t_solve` to get close to an
    equilibrium.'''
    t_span = (t, t + t_solve)
    return model._solver.solution_at_t_end(t_span, y_0)


def _objective(y_cur, solver, t, weights):
    '''Helper for `find`.'''
    y_new = solver.step(t, y_cur)
    diff = (y_new - y_cur) * weights
    return diff


def find(model, y_guess, t=0, t_solve=0, weights=1, **root_kwds):
    '''Find an equilibrium `y` while keeping
    `weighted_sum(y, weights)` constant.'''
    # Ensure `y_guess` is nonnegative.
    y_guess = numpy.clip(y_guess, 0, None)
    if t_solve > 0:
        y_guess = _solve(model, y_guess, t, t_solve)
    result = optimize.root(_objective, y_guess,
                           args=(model._solver, t, weights),
                           sparse=model._solver._sparse,
                           **root_kwds)
    assert result.success, result
    y = result.x
    # Scale `y` so that `weighted_sum()` is the same as for
    # `y_guess`.
    y *= (numerical.weighted_sum(y_guess, weights)
          / numerical.weighted_sum(y, weights))
    return y


def eigenvalues(model, equilibrium, t=0, k=5):
    '''Get the eigenvalues of `equilibrium`.'''
    n = len(equilibrium)
    J = model._solver.jacobian(t, equilibrium, equilibrium)
    evals = linalg.eigs(J, k=k, which='LR', return_eigenvectors=False)
    return numerical.sort_by_real_part(evals)
