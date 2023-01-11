'''Implementation of solvers for differential equations.'''

import abc

import numpy
import scipy.optimize

from . import _solution
from .. import _utility


_METHOD_DEFAULT = 'Crank–Nicolson'


class Solver(metaclass=abc.ABCMeta):
    '''Base class for solvers.'''

    @property
    @abc.abstractmethod
    def method(self):
        '''The name of the method.'''

    @classmethod
    def create(cls, func, method=_METHOD_DEFAULT, **kwds):
        '''Factory to choose the right solver class for `method`.'''
        for subcls in _utility.all_subclasses(cls):
            if subcls.method == method:
                return subcls(func, **kwds)
        raise ValueError(f'Unknown {method=}!')

    def __init__(self, func, states=None):
        self._func = func
        self.states = states

    def func(self, t, y):
        '''The result of `self._func(t, y)` as a `numpy.array()`.'''
        return numpy.asarray(self._func(t, y))

    @abc.abstractmethod
    def _step(self, t_cur, y_cur, t_new, y_new):
        '''Do a step.'''

    def __call__(self, t, y_0, y=None, _solution_wrap=True):
        '''Solve. `y` is storage for the solution, which will be built if not
        provided. `_solution_wrap=False` skips wrapping the solution in
        `_solution.Solution()` for speed.'''
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
        y[0] = y_0
        for k in range(1, len(t)):
            self._step(t[k - 1], y[k - 1], t[k], y[k])
        if _solution_wrap:
            return _solution.Solution(y, t, states=self.states)
        else:
            return y


class Euler(Solver):
    method = 'Euler'

    def _step(self, t_cur, y_cur, t_new, y_new):
        y_new[:] = y_cur + (t_new - t_cur) * self.func(t_cur, y_cur)


class _ImplicitSolver(Solver):
    @abc.abstractmethod
    def _const(self, t_cur, y_cur, t_new):
        '''Compute the term in `_objective()` that is independent of
        `y_new`.'''

    @abc.abstractmethod
    def _objective(self, y_new, t_cur, t_new, const):
        '''The objective function for `scipy.optimize.root()` in `_step()`.'''

    def _step(self, t_cur, y_cur, t_new, y_new):
        const = self._const(t_cur, y_cur, t_new)
        result = scipy.optimize.root(self._objective, y_cur,
                                     args=(t_cur, t_new, const))
        assert result.success, f't={t_new}: {result}'
        y_new[:] = result.x


class ImplicitEuler(_ImplicitSolver):
    method = 'Implicit Euler'

    def _const(self, t_cur, y_cur, t_new):
        '''Compute the term in `_objective()` that is independent of
        `y_new`.'''
        return y_cur

    def _objective(self, y_new, t_cur, t_new, const):
        return y_new - (t_new - t_cur) * self.func(t_new, y_new) - const


class CrankNicolson(_ImplicitSolver):
    method = 'Crank–Nicolson'

    @staticmethod
    def _t_mid(t_cur, t_new):
        return 0.5 * (t_new + t_cur)

    def _const(self, t_cur, y_cur, t_new):
        '''Compute the term in `_objective()` that is independent of
        `y_new`.'''
        t_mid = self._t_mid(t_cur, t_new)
        return y_cur + 0.5 * (t_new - t_cur) * self.func(t_mid, y_cur)

    def _objective(self, y_new, t_cur, t_new, const):
        t_mid = self._t_mid(t_cur, t_new)
        return y_new - 0.5 * (t_new - t_cur) * self.func(t_mid, y_new) - const


def solve(func, t, y_0, y=None, _solution_wrap=True, **kwds):
    '''Solve the ODE defined by the derivatives in `func`.'''
    solver = Solver.create(func, **kwds)
    return solver(t, y_0, y=y, _solution_wrap=_solution_wrap)
