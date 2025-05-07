'''Find equilibria.'''

import numpy

from .. import _utility
from .._utility import _transform


def solution_after_t_solve(model, t_0, t_solve, y_0, **kwds):
    '''Solve from `t_0` to `t_0 + t_solve` to get closer to an
    equilibrium or limit cycle.'''
    assert t_solve >= 0
    if t_solve == 0:
        return (t_0, y_0)
    t_1 = t_0 + t_solve
    y_1 = model.solver.solution_at_t_end((t_0, t_1), y_0, **kwds)
    return (t_1, y_1)


def _root_objective(x_cur, t, solver, transform):
    '''Helper for `find(..., solver='root', ...)`.'''
    y_cur = transform.inverse(x_cur)
    y_new = solver.step(t, y_cur)
    x_new = transform(y_new)
    return x_new - x_cur


def _fixed_point_objective(x_cur, t, solver, transform):
    '''Helper for `find(..., solver='fixed_point', ...)`.'''
    y_cur = transform.inverse(x_cur)
    y_new = solver.step(t, y_cur)
    x_new = transform(y_new)
    return x_new


# pylint: disable-next=too-many-arguments
def find(model, y_guess, t=0, *,
         t_solve=0, solver='root', weights=1, display=False,
         **kwds):
    '''Find an equilibrium `y` while keeping
    `weighted_sum(y, weights)` constant.'''
    # Ensure `y_guess` is nonnegative.
    y_guess = numpy.clip(y_guess, 0, None)
    (t, y_guess) = solution_after_t_solve(model, t, t_solve, y_guess,
                                          display=display)
    total = _utility.numerical.weighted_sum(y_guess, weights)
    weights_total = weights / total
    transform = _transform.Simplex(weights=weights_total)
    x_guess = transform(y_guess)
    if solver == 'root':
        x = _utility.optimize.root(_root_objective, x_guess,
                                   args=(t, model.solver, transform),
                                   sparse=model.solver.sparse,
                                   display=display,
                                   **kwds)
    elif solver == 'fixed_point':
        x = _utility.optimize.fixed_point(_fixed_point_objective, x_guess,
                                          args=(t, model.solver, transform),
                                          **kwds)
    else:
        raise ValueError(f'Unknown {solver=}!')
    y = transform.inverse(x)
    assert numpy.isclose(
        _utility.numerical.weighted_sum(y, weights),
        total
    )
    return y


def eigenvalues(model, equilibrium, t=0, k=5, verbose=False):
    '''Get the `k` eigenvalues of `equilibrium` with largest real
    part.'''
    if verbose:
        print('Building the jacobian...')
    jac = model.solver.jacobian(t, equilibrium, equilibrium)
    if verbose:
        print('Finding eigenvalues...')
    evals = _utility.linalg.eigs(jac, k=k, which='LR',
                                 return_eigenvectors=False)
    return evals
