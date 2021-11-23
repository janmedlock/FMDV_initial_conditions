'''Implementation of solvers for differential equations.'''

import abc

import numpy
import scipy.optimize

from . import solution
from . import utility


_METHOD_DEFAULT = 'Crank–Nicolson'


class Solver(metaclass=abc.ABCMeta):
    '''Base class for solvers.'''

    @property
    @abc.abstractmethod
    def method(self):
        '''The name of the method.'''

    @classmethod
    def create(cls, func, method=_METHOD_DEFAULT, states=None):
        '''Factory to choose the right solver class for `method`.'''
        for subcls in utility.all_subclasses(cls):
            if subcls.method == method:
                return subcls(func, states=states)
        raise ValueError(f'Unknown {method=}!')

    def __init__(self, func, states=None):
        self._func = func
        self.states = states

    def func(self, t, y):
        '''The result of `self._func(t, y)` as a `numpy.array()`.'''
        return numpy.asarray(self._func(t, y))

    @abc.abstractmethod
    def _y_new(self, t_new, t_cur, y_cur):
        '''Do a step.'''

    def __call__(self, t, y_0, y=None, _solution=True):
        '''Solve.'''
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
        y[0] = y_0
        for k in range(1, len(t)):
            y[k] = self._y_new(t[k], t[k - 1], y[k - 1])
        if _solution:
            return solution.Solution(t, y, states=self.states)
        else:
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
    return solver(t, y_0)
