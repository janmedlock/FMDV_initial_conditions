'''Find limit cycles.'''

import numpy

from . import _equilibrium, _fundamental, _poincare
from .. import _utility


def check(y, weights=1, **kwds):
    '''Check whether `(t, y)` is a limit cycle.'''
    assert y.ndim == 2
    return numpy.allclose(y[0] * weights,
                          y[-1] * weights,
                          **kwds)


# pylint: disable-next=too-many-arguments
def find_with_period(model, period, t_0, y_0_guess, *,
                     t_solve=0, weights=1, solution=True, display=False,
                     **kwds):
    '''Find a limit cycle with period `period` while keeping
    `weighted_sum(y_0, weights)` constant.'''
    # Ensure `y_guess` is nonnegative.
    y_0_guess = numpy.clip(y_0_guess, 0, None)
    (t_0, y_0_guess) = _equilibrium.solution_after_t_solve(model,
                                                           t_0, t_solve,
                                                           y_0_guess,
                                                           display=display)
    poincare_map = _poincare.Map(model, period, t_0)
    y_0 = poincare_map.find_fixed_point(y_0_guess,
                                        weights=weights,
                                        display=display,
                                        **kwds)
    if solution:
        # Return the solution at the `t` values, not just at the end time.
        (t, y) = poincare_map.solve(y_0, display=display)
        assert check(y, weights=weights)
        return (t, y)
    return y_0


# pylint: disable-next=too-many-arguments
def find_subharmonic(model, period_0, t_0, y_0_guess, *,
                     order_max=10, t_solve=0, display=False,
                     **kwds):
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
                                    display=display,
                                    **kwds)
        except RuntimeError:
            pass
    msg = f'No subharmonic limit cycle found with order <= {order_max}'
    raise RuntimeError(msg)


def monodromy(model, limit_cycle, display=False):
    '''Get the monodromy matrix.'''
    # pylint: disable-next=invalid-name
    Psi = _fundamental.monodromy(model, limit_cycle, display=display)
    return Psi


def characteristic_multipliers(model, limit_cycle, k=5, display=False):
    '''Get the `k` characteristic multipliers of `limit_cycle` with
    largest magnitude.'''
    # pylint: disable-next=invalid-name
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
