'''Solver.'''

import numpy
import scipy.optimize
import scipy.sparse

from . import _solution
from .. import _utility


class Solver:
    '''Crank–Nicolson solver.'''

    def __init__(self, model):
        self.model = model

    def func(self, t, y):
        '''The result of `self.model(t, y)` as a `numpy.array()`.'''
        return numpy.asarray(self.model(t, y))

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

    def _step(self, t_cur, y_cur, t_new, y_new):
        '''Do a step.'''
        const = self._const(t_cur, y_cur, t_new)
        result = scipy.optimize.root(self._objective, y_cur,
                                     args=(t_cur, t_new, const))
        assert result.success, f't={t_new}: {result}'
        y_new[:] = result.x

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
            return _solution.Solution(y, t, states=self.model.states)
        else:
            return y


def solve(model, t, y_0, y=None, _solution_wrap=True):
    '''Solve the ODE defined by the derivatives in `model`.'''
    solver = Solver(model)
    return solver(t, y_0, y=y, _solution_wrap=_solution_wrap)
