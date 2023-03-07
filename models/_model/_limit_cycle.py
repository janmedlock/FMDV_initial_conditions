'''Find limit cycles.'''

import numpy

from . import _equilibrium, _fundamental, _poincaré
from .. import _utility


def _objective(y_0_cur, poincaré_map, weights, display):
    '''Helper for `find_with_period`.'''
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
    result = _utility.optimize.root(_objective, y_0_guess,
                                    args=(poincaré_map, weights, display),
                                    sparse=model._solver._sparse,
                                    display=display,
                                    **root_kwds)
    assert result.success, result
    y_0 = result.x
    # Scale `y_0` so that `weighted_sum()` is the same as for
    # `y_0_guess`.
    y_0 *= (_utility.numerical.weighted_sum(y_0_guess, weights)
            / _utility.numerical.weighted_sum(y_0, weights))
    if solution:
        # Return the solution at the `t` values, not just at the end time.
        return poincaré_map.solve(y_0, display=display)
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
    '''Get the characteristic multipliers of `limit_cycle`.'''
    Psi = monodromy(model, limit_cycle, display=display)
    mlts = _utility.linalg.eigs(Psi, k=k, which='LM',
                                return_eigenvectors=False)
    # One of the multipliers of a limit cycle is always 1, so drop
    # that.
    drop = numpy.abs(mlts - 1).argmin()
    if numpy.isclose(mlts[drop], 1):
        mlts = numpy.delete(mlts, drop)
    return _utility.numerical.sort_by_abs(mlts)


def characteristic_exponents(model, limit_cycle, k=5, display=False):
    '''Get the characteristic exponents of `limit_cycle`.'''
    mlts = characteristic_multipliers(model, limit_cycle,
                                      k=k, display=display)
    t = limit_cycle.index
    period = t[-1] - t[0]
    exps = numpy.log(mlts) / period
    return exps
