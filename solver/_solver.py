'''Implementation of solvers for differential equations.'''

import abc

import numpy
import scipy.optimize

from . import _utility


_METHOD_DEFAULT = 'Crank–Nicolson'


class Solver(metaclass=abc.ABCMeta):
    '''Base class for solvers.'''

    @property
    @abc.abstractmethod
    def method(self):
        '''The name of the method.'''

    @classmethod
    def create(cls, func, method=_METHOD_DEFAULT):
        '''Factory to choose the right solver class for `method`.'''
        for subcls in _utility.all_subclasses(cls):
            if subcls.method == method:
                return subcls(func)
        raise ValueError(f'Unknown {method=}!')

    def __init__(self, func):
        self._func = func

    def func(self, t, y):
        '''The result of `self._func(t, y)` as a `numpy.array()`.'''
        return numpy.asarray(self._func(t, y))

    @abc.abstractmethod
    def _y_new(self, t_new, t_cur, y_cur):
        '''Do a step.'''

    def solve(self, t, y_0=None, y=None):
        '''Solve.'''
        if ((y_0 is None and y is None)
                or (y_0 is not None and y is not None)):
            raise ValueError('One of `y_0` or `y` must be specified.')
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
            y[0] = y_0
        for k in range(1, len(t)):
            y[k] = self._y_new(t[k], t[k - 1], y[k - 1])
        return y


class Euler(Solver):
    method = 'Euler'

    def _y_new(self, t_new, t_cur, y_cur):
        return y_cur + (t_new - t_cur) * self.func(t_cur, y_cur)


class _ImplicitSolver(Solver):
    @property
    @abc.abstractmethod
    def _a(self):
        '''Coefficient in `_objective`.'''

    @abc.abstractmethod
    def _b(self, t_new, t_cur, y_cur):
        '''Compute the term in `_objective()` that is independent of
        `y_new`.'''

    def _objective(self, y_new, t_new, t_cur, b):
        return y_new - self._a * (t_new - t_cur) * self.func(t_new, y_new) - b

    def _y_new(self, t_new, t_cur, y_cur):
        result = scipy.optimize.root(self._objective, y_cur,
                                     args=(t_new, t_cur,
                                           self._b(t_new, t_cur, y_cur)))
        assert result.success, f't={t_new}: {result}'
        return result.x


class ImplicitEuler(_ImplicitSolver):
    method = 'Implicit Euler'

    _a = 1

    def _b(self, t_new, t_cur, y_cur):
        '''Compute the term in `_objective()` that is independent of
        `y_new`.'''
        return y_cur


class CrankNicolson(_ImplicitSolver):
    method = 'Crank–Nicolson'

    _a = 0.5

    def _b(self, t_new, t_cur, y_cur):
        '''Compute the term in `_objective()` that is independent of
        `y_new`.'''
        return y_cur + 0.5 * (t_new - t_cur) * self.func(t_cur, y_cur)


def solve(func, t, y_0, **kwds):
    '''Solve the ODE defined by the derivatives in `func`.'''
    solver = Solver.create(func, **kwds)
    return solver.solve(t, y_0=y_0)
