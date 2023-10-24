'''Find equilibria.'''

import numpy

from .. import _utility


def solution_after_t_solve(model, t_0, t_solve, y_0, **kwds):
    '''Solve from `t_0` to `t_0 + t_solve` to get closer to an
    equilibrium or limit cycle.'''
    assert t_solve >= 0
    if t_solve == 0:
        return (t_0, y_0)
    t_1 = t_0 + t_solve
    y_1 = model._solver.solution_at_t_end((t_0, t_1), y_0, **kwds)
    return (t_1, y_1)


def _objective(y_cur, solver, t, weights):
    '''Helper for `find`.'''
    y_new = solver.step(t, y_cur)
    diff = (y_new - y_cur) * weights
    return diff


def find(model, y_guess, t=0, t_solve=0, weights=1,
         display=False, **root_kwds):
    '''Find an equilibrium `y` while keeping
    `weighted_sum(y, weights)` constant.'''
    # Ensure `y_guess` is nonnegative.
    y_guess = numpy.clip(y_guess, 0, None)
    (t, y_guess) = solution_after_t_solve(model, t, t_solve, y_guess,
                                          display=display)
    result = _utility.optimize.root(_objective, y_guess,
                                    args=(model._solver, t, weights),
                                    sparse=model._solver._sparse,
                                    display=display,
                                    **root_kwds)
    assert result.success, result
    y = result.x
    # Scale `y` so that `weighted_sum()` is the same as for
    # `y_guess`.
    y *= (_utility.numerical.weighted_sum(y_guess, weights)
          / _utility.numerical.weighted_sum(y, weights))
    return y


def eigenvalues(model, equilibrium, t=0, k=5, verbose=False):
    '''Get the `k` eigenvalues of `equilibrium` with largest real
    part.'''
    if verbose:
        print('Building the jacobian...')
    jac = model._solver.jacobian(t, equilibrium, equilibrium)
    if verbose:
        print('Finding eigenvalues...')
    evals = _utility.linalg.eigs(jac, k=k, which='LR',
                                 return_eigenvectors=False)
    return evals
