'''Solvers for differential equations.'''

import numpy
import scipy.optimize

import utility


def euler(func, t, t_step, y):
    '''Solve the ODE defined by the derivatives in `func` using the
    explicit Euler scheme.'''
    for k in range(1, len(t)):
        y[k] = y[k - 1] + t_step * func(t[k - 1], y[k - 1])
        assert numpy.all(numpy.isfinite(y[k])), \
            f't[{k}]={t[k]}: y[{k}]={y[k]}'


def _c(y, t, func, d):
    '''Compute the `c` term for `_objective()` that is independent of
    `y_k`.'''
    if d == 0:
        return y
    else:
        return y + d * func(t, y)


def _objective(y_k, t_k, func, b, c):
    '''Function to solve in each step.'''
    return y_k - b * func(t_k, y_k) - c


def _implicit(func, t, y, b, d):
    '''Solve the ODE defined by the derivatives in `func` using an
    implicit scheme.'''
    for k in range(1, len(t)):
        c = _c(y[k - 1], t[k - 1], func, d)
        result = scipy.optimize.root(_objective,
                                     y[k - 1],
                                     args=(t[k], func, b, c))
        assert result.success, f't[{k}]={t[k]}: {result}'
        assert numpy.all(numpy.isfinite(result.x)), \
            f't[{k}]={t[k]}: y[{k}]={result.x}'
        y[k] = result.x


def implicit_euler(func, t, t_step, y):
    '''Solve the ODE defined by the derivatives in `func` using the
    implicit Euler scheme.'''
    _implicit(func, t, y, t_step, 0)


def crank_nicolson(func, t, t_step, y):
    '''Solve the ODE defined by the derivatives in `func` using the
    Crank–Nicolson scheme.'''
    _implicit(func, t, y, t_step / 2, t_step / 2)


def solver(func, t_start, t_end, t_step, y_start, method='crank_nicolson'):
    '''Solve the ODE defined by the derivatives in `func`.'''
    t = utility.arange(t_start, t_end, t_step)
    y = numpy.empty((len(t), len(y_start)))
    y[0] = y_start
    if method == 'euler':
        euler(func, t, t_step, y)
    elif method == 'implicit_euler':
        implicit_euler(func, t, t_step, y)
    elif method == 'crank_nicolson':
        crank_nicolson(func, t, t_step, y)
    else:
        raise ValueError(f'Unknown {method=}!')
    return (t, y)
