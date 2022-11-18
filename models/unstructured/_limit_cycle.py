'''Find limit cycles.'''

import numpy
import scipy.optimize

from . import _fundamental
from . import _poincaré
from .. import _utility


def _poincaré_map_transform(x, poincaré_map, transform, y):
    '''Find the transformed state after one period.'''
    y_0 = transform.inverse(x)
    poincaré_map(y_0, y=y, _solution_wrap=False)
    return transform(y[-1])


def find_with_period(func, period, t_0, t_step, y_0_guess,
                     _solution_wrap=True, **kwds):
    '''Find a limit cycle with period `period`.'''
    poincaré_map = _poincaré.Map(func, period, t_0, t_step, **kwds)
    # Storage for intermediate y values.
    y = poincaré_map.build_y(y_0_guess)
    # Find the `y_0` value that is a fixed point of the Poincaré map,
    # i.e. that gives `y(t_0 + period) = y_0`, while keeping
    # `y_0.sum()` constant.
    transform = _utility.TransformConstantSum(y_0_guess)
    x = scipy.optimize.fixed_point(_poincaré_map_transform,
                                   transform(y_0_guess),
                                   args=(poincaré_map, transform, y))
    y_0 = transform.inverse(x)
    # Return the solution at the intermediate t values.
    return poincaré_map.solve(y_0, y=y, _solution_wrap=_solution_wrap)


def find_subharmonic(func, period_0, t_0, t_step, y_0_guess,
                     order_max=10, _solution_wrap=True, **kwds):
    '''Find a subharmonic limit cycle for a system with forcing period
    `period_0`.'''
    for order in numpy.arange(1, order_max + 1):
        try:
            return find_with_period(func, order * period_0,
                                    t_0, t_step, y_0_guess,
                                    _solution_wrap=_solution_wrap,
                                    **kwds)
        except RuntimeError:
            pass
    msg = f'No subharmonic limit cycle found with order <= {order_max}'
    raise RuntimeError(msg)


def monodromy_matrix(func, limit_cycle, **kwds):
    '''Get the monodromy matrix.'''
    phi = _fundamental.solution(func, limit_cycle, **kwds)
    return phi[-1]


def characteristic_multipliers(func, limit_cycle, **kwds):
    '''Get the characteristic multipliers of `limit_cycle`.'''
    psi = monodromy_matrix(func, limit_cycle, **kwds)
    mlts = numpy.linalg.eigvals(psi)
    # Drop the one closest to 1.
    drop = numpy.abs(mlts - 1).argmin()
    assert numpy.isclose(mlts[drop], 1)
    mlts = numpy.delete(mlts, drop)
    return _utility.sort_by_abs(mlts)


def characteristic_exponents(func, limit_cycle, **kwds):
    '''Get the characteristic exponents of `limit_cycle`.'''
    mlts = characteristic_multipliers(func, limit_cycle, **kwds)
    t = limit_cycle.index
    period = t[-1] - t[0]
    exps = numpy.log(mlts) / period
    return exps
