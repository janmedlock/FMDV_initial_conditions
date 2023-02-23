'''Solver.'''

import numpy
import scipy.optimize

from .. import _solver
from .. import _utility


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    def __init__(self, model):
        self.t_step = model.t_step
        super().__init__(model)

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
        '''Helper for `.step()`.'''
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


def solve(model, t_span, y_0,
          t=None, y=None, display=False):
    '''Solve the model.'''
    solver = Solver(model)
    return solver.solve(t_span, y_0,
                        t=t, y=y, display=display)
