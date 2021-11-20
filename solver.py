'''Solvers for differential equations.'''

import abc

import numpy
import scipy.optimize

import utility


class _Solver(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def method(self):
        '''The name of the method.'''

    @classmethod
    def create(cls, method, func, t, y_0):
        '''Factory to choose the right solver class for `method`.'''
        for subcls in utility.all_subclasses(cls):
            if subcls.method == method:
                return subcls(func, t, y_0)
        raise ValueError(f'Unknown {method=}!')

    def __init__(self, func, t, y_0):
        self._func = func
        self.t = t
        self.y_0 = y_0

    def func(self, t, y):
        '''The result of `self._func(t, y)` as a `numpy.array()`.'''
        return numpy.asarray(self._func(t, y))

    @abc.abstractmethod
    def _y_new(self, t_new, t_cur, y_cur):
        '''Do a step.'''

    def solve(self):
        '''Solve.'''
        shape = (len(self.t), *numpy.shape(self.y_0))
        y = numpy.empty(shape)
        y[0] = self.y_0
        for k in range(1, len(self.t)):
            y[k] = self._y_new(self.t[k], self.t[k - 1], y[k - 1])
        return y


class _Euler(_Solver):
    method = 'Euler'

    def _y_new(self, t_new, t_cur, y_cur):
        return y_cur + (t_new - t_cur) * self.func(t_cur, y_cur)


class _ImplicitSolver(_Solver):
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


class _ImplicitEuler(_ImplicitSolver):
    method = 'Implicit Euler'

    _a = 1

    def _b(self, t_new, t_cur, y_cur):
        '''Compute the term in `_objective()` that is independent of
        `y_new`.'''
        return y_cur


class _CrankNicolson(_ImplicitSolver):
    method = 'Crank–Nicolson'

    _a = 0.5

    def _b(self, t_new, t_cur, y_cur):
        '''Compute the term in `_objective()` that is independent of
        `y_new`.'''
        return y_cur + 0.5 * (t_new - t_cur) * self.func(t_cur, y_cur)


def solve(func, t, y_0, method='Crank–Nicolson'):
    '''Solve the ODE defined by the derivatives in `func`.'''
    solver = _Solver.create(method, func, t, y_0)
    return solver.solve()
