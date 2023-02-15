'''Solver.'''

import numpy
import scipy.optimize

from .. import _utility


class Solver:
    '''Crank–Nicolson solver.'''

    def __init__(self, model, time_step):
        self.model = model
        self.time_step = time_step
        self._build_matrices()

    def _beta(self):
        beta = (self.model.transmission.rate
                * numpy.array((0, 0, 0, 1, 0)))
        return beta

    def _H(self):
        n = len(self.model.states)
        H = numpy.identity(n)
        return H

    def _F(self):
        mu = self.model.death_rate_mean
        omega = 1 / self.model.waning.mean
        rho = 1 / self.model.progression.mean
        gamma = 1 / self.model.recovery.mean
        F = numpy.array([
            [- (omega + mu), 0, 0, 0, 0],
            [omega, - mu, 0, 0, 0],
            [0, 0, - (rho + mu), 0, 0],
            [0, 0, rho, - (gamma + mu), 0],
            [0, 0, 0, gamma, - mu]
        ])
        return F

    @staticmethod
    def _T():
        T = numpy.array([
            [0, 0, 0, 0, 0],
            [0, - 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        return T

    @staticmethod
    def _B():
        B = numpy.array([
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        return B

    def _build_matrices(self):
        n = len(self.model.states)
        self.beta = self._beta()
        self.H = self._H()
        self.F = self._F()
        self.T = self._T()
        self.B = self._B()

    def _objective(self, y_new, HFB0, HFBTy1):
        lambdaT0 = (self.beta @ y_new) * self.T
        HFBT0 = HFB0 - self.time_step / 2 * lambdaT0
        return HFBT0 @ y_new - HFBTy1

    def _step(self, t_cur, y_cur, y_new):
        '''Do a step.'''
        lambdaT1 = (self.beta @ y_cur) * self.T
        t_mid = t_cur + 0.5 * self.time_step
        bB = self.model.birth.rate(t_mid) * self.B
        HFB0 = self.H - self.time_step / 2 * (self.F + bB)
        HFBT1 = self.H + self.time_step / 2 * (self.F + lambdaT1 + bB)
        HFBTy1 = HFBT1 @ y_cur
        result = scipy.optimize.root(self._objective, y_cur,
                                     args=(HFB0, HFBTy1))
        assert result.success, f't={t_cur}: {result}'
        y_new[:] = result.x

    def __call__(self, t_span, y_0,
                 t=None, y=None, _solution_wrap=True):
        '''Solve. `y` is storage for the solution, which will be built if not
        provided. `_solution_wrap=False` skips wrapping the solution in
        `model.Solution()` for speed.'''
        if t is None:
            t = _utility.build_t(*t_span, self.time_step)
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
        y[0] = y_0
        for ell in range(1, len(t)):
            self._step(t[ell - 1], y[ell - 1], y[ell])
        if _solution_wrap:
            return self.model.Solution(y, t)
        else:
            return y


def solve(model, t_span, time_step, y_0,
          t=None, y=None, _solution_wrap=True):
    '''Solve the `model` system of ODEs.'''
    solver = Solver(model, time_step)
    return solver(t_span, y_0,
                  t=t, y=y, _solution_wrap=_solution_wrap)
