'''Solver.'''

import numpy
import scipy.optimize

from .. import _utility


class Solver:
    '''Crank–Nicolson solver.'''

    def __init__(self, model):
        self.model = model
        self.t_step = self.model.t_step
        self._build_matrices()
        self._check_matrices()

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
            [- omega - mu, 0, 0, 0, 0],
            [omega, - mu, 0, 0, 0],
            [0, 0, - rho - mu, 0, 0],
            [0, 0, rho, - gamma - mu, 0],
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

    def _check_matrices(self):
        assert _utility.is_nonnegative(self.beta)
        assert _utility.is_Z_matrix(self.H)
        assert _utility.is_nonnegative(self.H)
        assert _utility.is_Metzler_matrix(self.F)
        assert _utility.is_Metzler_matrix(self.T)
        assert _utility.is_Metzler_matrix(self.B)
        assert _utility.is_nonnegative(self.B)
        HFB0 = (self.H
                - self.t_step / 2 * (self.F
                                     + self.model.birth.rate_max * self.B))
        assert _utility.is_M_matrix(HFB0)
        HFB1 = (self.H
                + self.t_step / 2 * (self.F
                                     + self.model.birth.rate_min * self.B))
        assert _utility.is_nonnegative(HFB1)

    def _objective(self, y_new, HFB0, HFBTy1):
        lambdaT0 = (self.beta @ y_new) * self.T
        HFBT0 = HFB0 - self.t_step / 2 * lambdaT0
        return HFBT0 @ y_new - HFBTy1

    def step(self, t_cur, y_cur, display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + 0.5 * self.t_step
        bB = self.model.birth.rate(t_mid) * self.B
        HFB0 = self.H - self.t_step / 2 * (self.F + bB)
        lambdaT1 = (self.beta @ y_cur) * self.T
        HFBT1 = self.H + self.t_step / 2 * (self.F + lambdaT1 + bB)
        HFBTy1 = HFBT1 @ y_cur
        result = scipy.optimize.root(self._objective, y_cur,
                                     args=(HFB0, HFBTy1))
        assert result.success, f'{t_cur=}\n{result=}'
        y_new = result.x
        return y_new

    def solve(self, t_span, y_0,
              t=None, y=None, display=False, _solution_wrap=True):
        '''Solve. `y` is storage for the solution, which will be built if not
        provided. `_solution_wrap=False` skips wrapping the solution in
        `model.Solution()` for speed.'''
        if t is None:
            t = _utility.build_t(*t_span, self.t_step)
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
        y[0] = y_0
        for ell in range(1, len(t)):
            y[ell] = self.step(t[ell - 1], y[ell - 1], display=display)
        if _solution_wrap:
            return self.model.Solution(y, t)
        else:
            return y

    def _solution_at_t_end(self, t_span, y_0,
                           t=None, y_temp=None,
                           _solution_wrap=True):
        '''Find the value of the solution at `t_span[1]`.'''
        if t is None:
            t = _utility.build_t(*t_span, self.t_step)
        if y_temp is None:
            y_temp = numpy.empty((2, *numpy.shape(y_0)))
        (y_cur, y_new) = y_temp
        y_new[:] = y_0
        for t_cur in t[:-1]:
            # Update so that what was the new value of the solution is
            # now the current value and what was the current value of
            # the solution will be storage space for the new value.
            (y_cur, y_new) = (y_new, y_cur)
            y_new[:] = self.step(t_cur, y_cur)
        if _solution_wrap:
            return self.model.Solution(y_new, t)
        else:
            return y_new


def solution(model, t_span, y_0,
             t=None, y=None, display=False, _solution_wrap=True):
    '''Solve the model.'''
    solver = Solver(model)
    return solver.solve(t_span, y_0,
                        t=t, y=y, display=display,
                        _solution_wrap=_solution_wrap)
