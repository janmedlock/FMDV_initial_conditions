'''Find limit cycles.'''

import numpy
import scipy.optimize

from . import _fundamental
from . import _poincaré
from .. import _utility


def _objective(y_0, poincaré_map, weights, y):
    '''Helper for `find_with_period`.'''
    poincaré_map(y_0, y=y, _solution_wrap=False)
    return (y[-1] - y_0) * weights


def find_with_period(model, period, t_0, y_0_guess,
                     weights=1, _solution_wrap=True,
                     **root_kwds):
    '''Find a limit cycle with period `period` while keeping
    `weighted_sum(y_0, weights)` constant.'''
    poincaré_map = _poincaré.Map(model, period, t_0)
    # Storage for intermediate y values.
    y = poincaré_map.build_y(y_0_guess)
    # Find a fixed point `y_0` of the Poincaré map, i.e. that gives
    # `y(t_0 + period) = y_0`.
    y_0_total = _utility.weighted_sum(y_0_guess, weights)
    result = scipy.optimize.root(_objective, y_0_guess,
                                 args=(poincaré_map, weights, y),
                                 **root_kwds)
    assert result.success, result
    y_0 = result.x
    # Scale `y_0` so that `weighted_sum()` is the same as for
    # `y_0_guess`.
    y_0 *= (_utility.weighted_sum(y_0_guess, weights)
            / _utility.weighted_sum(y_0, weights))
    # Return the solution at the `t` values, not just at the end time.
    return poincaré_map.solve(y_0, y=y, _solution_wrap=_solution_wrap)


def find_subharmonic(model, period_0, t_0, y_0_guess,
                     order_max=10, weights=1,
                     _solution_wrap=True, **root_kwds):
    '''Find a subharmonic limit cycle for a system with forcing period
    `period_0`.'''
    for order in numpy.arange(1, order_max + 1):
        try:
            return find_with_period(model, order * period_0,
                                    t_0, y_0_guess,
                                    weights=weights,
                                    _solution_wrap=_solution_wrap,
                                    **root_kwds)
        except RuntimeError:
            pass
    msg = f'No subharmonic limit cycle found with order <= {order_max}'
    raise RuntimeError(msg)


def monodromy_matrix(model, limit_cycle):
    '''Get the monodromy matrix.'''
    phi = _fundamental.solution(model, limit_cycle)
    return phi[-1]


def characteristic_multipliers(model, limit_cycle):
    '''Get the characteristic multipliers of `limit_cycle`.'''
    psi = monodromy_matrix(model, limit_cycle)
    mlts = numpy.linalg.eigvals(psi)
    # Drop the one closest to 1.
    drop = numpy.abs(mlts - 1).argmin()
    assert numpy.isclose(mlts[drop], 1)
    mlts = numpy.delete(mlts, drop)
    return _utility.sort_by_abs(mlts)


def characteristic_exponents(model, limit_cycle):
    '''Get the characteristic exponents of `limit_cycle`.'''
    mlts = characteristic_multipliers(model, limit_cycle)
    t = limit_cycle.index
    period = t[-1] - t[0]
    exps = numpy.log(mlts) / period
    return exps
