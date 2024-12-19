'''Find limit cycles.'''

import numpy

from . import _equilibrium, _fundamental, _poincaré
from .. import _utility
from .._utility import _transform


def check(y, weights=1, **kwds):
    '''Check whether `(t, y)` is a limit cycle.'''
    assert y.ndim == 2
    return numpy.allclose(y[0] * weights,
                          y[-1] * weights,
                          **kwds)


def _objective(x_0_cur, poincaré_map, weights, transform, display):
    '''Helper for `find_with_period`.'''
    y_0_cur = transform.inverse(x_0_cur)
    y_0_new = poincaré_map(y_0_cur, display=display)
    diff = (y_0_new - y_0_cur) * weights
    return diff


def find_with_period(model, period, t_0, y_0_guess,
                     t_solve=0, weights=1, solution=True,
                     display=False, **root_kwds):
    '''Find a limit cycle with period `period` while keeping
    `weighted_sum(y_0, weights)` constant.'''
    # Ensure `y_guess` is nonnegative.
    y_0_guess = numpy.clip(y_0_guess, 0, None)
    (t_0, y_0_guess) = _equilibrium.solution_after_t_solve(model,
                                                           t_0, t_solve,
                                                           y_0_guess,
                                                           display=display)
    # Find a fixed point `y_0` of the Poincaré map, i.e. that gives
    # `y(t_0 + period) = y_0`.
    poincaré_map = _poincaré.Map(model, period, t_0)
    transform = _transform.Logarithm(a=1e-6,
                                     weights=weights)
    x_0_guess = transform(y_0_guess)
    result = _utility.optimize.root(
        _objective, x_0_guess,
        args=(poincaré_map, weights, transform, display),
        sparse=model._solver.sparse,
        display=display,
        **root_kwds
    )
    assert result.success, result
    y_0 = transform.inverse(result.x)
    # Scale `y_0` so that `weighted_sum()` is the same as for
    # `y_0_guess`.
    y_0 *= (_utility.numerical.weighted_sum(y_0_guess, weights)
            / _utility.numerical.weighted_sum(y_0, weights))
    if solution:
        # Return the solution at the `t` values, not just at the end time.
        (t, y) = poincaré_map.solve(y_0, display=display)
        assert check(y, weights=weights)
        return (t, y)
    else:
        return y_0


def find_subharmonic(model, period_0, t_0, y_0_guess,
                     t_solve=0, order_max=10, weights=1, solution=True,
                     display=False, **root_kwds):
    '''Find a subharmonic limit cycle for a system with forcing period
    `period_0`.'''
    # Ensure `y_guess` is nonnegative.
    y_0_guess = numpy.clip(y_0_guess, 0, None)
    (t_0, y_0_guess) = _equilibrium.solution_after_t_solve(model,
                                                           t_0, t_solve,
                                                           y_0_guess,
                                                           display=display)
    for order in numpy.arange(1, order_max + 1):
        try:
            return find_with_period(model, order * period_0,
                                    t_0, y_0_guess,
                                    weights=weights,
                                    solution=solution,
                                    display=display,
                                    **root_kwds)
        except RuntimeError:
            pass
    msg = f'No subharmonic limit cycle found with order <= {order_max}'
    raise RuntimeError(msg)


def monodromy(model, limit_cycle, display=False):
    '''Get the monodromy matrix.'''
    Psi = _fundamental.monodromy(model, limit_cycle, display=display)
    return Psi


def characteristic_multipliers(model, limit_cycle, k=5, display=False):
    '''Get the `k` characteristic multipliers of `limit_cycle` with
    largest magnitude.'''
    Psi = monodromy(model, limit_cycle, display=display)
    mlts = _utility.linalg.eigs(Psi, k=k, which='LM',
                                return_eigenvectors=False)
    return mlts


def characteristic_exponents(model, limit_cycle, k=5, display=False):
    '''Get the characteristic exponents of `limit_cycle`.'''
    mlts = characteristic_multipliers(model, limit_cycle,
                                      k=k, display=display)
    t = limit_cycle.index
    period = t[-1] - t[0]
    exps = numpy.log(mlts) / period
    return exps
