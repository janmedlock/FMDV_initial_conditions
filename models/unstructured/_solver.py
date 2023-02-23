'''Solver.'''

import numpy
import scipy.linalg
import scipy.optimize

from .. import _solver
from .. import _utility


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    def __init__(self, model):
        self.t_step = model.t_step
        super().__init__(model)

    def _I(self):
        n = len(self.model.states)
        I = numpy.identity(n)
        return I

    def _beta(self):
        beta = (self.model.transmission.rate
                * numpy.array((0, 0, 0, 1, 0)))
        return beta

    def _H(self):
        H = self.I
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
        self.I = self._I()
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
        HFB_new = (self.H
                   - self.t_step / 2 * (self.F
                                        + self.model.birth.rate_max * self.B))
        assert _utility.is_M_matrix(HFB_new)
        HFB_cur = (self.H
                   + self.t_step / 2 * (self.F
                                        + self.model.birth.rate_min * self.B))
        assert _utility.is_nonnegative(HFB_cur)

    def _objective(self, y_new, HFB_new, HFTBy_cur):
        '''Helper for `.step()`.'''
        HFTB_new = (HFB_new
                    - self.t_step / 2 * self.beta @ y_new * self.T)
        return HFTB_new @ y_new - HFTBy_cur

    def step(self, t_cur, y_cur, display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.birth.rate(t_mid)
        HFB_new = (self.H
                   - self.t_step / 2 * (self.F
                                        + b_mid * self.B))
        HFTB_cur = (self.H
                    + self.t_step / 2 * (self.F
                                         + self.beta @ y_cur * self.T
                                         + b_mid * self.B))
        HFTBy_cur = HFTB_cur @ y_cur
        y_new_guess = y_cur
        result = scipy.optimize.root(self._objective, y_new_guess,
                                     args=(HFB_new, HFTBy_cur))
        assert result.success, f'{t_cur=}\n{result=}'
        y_new = result.x
        return y_new

    def jacobian(self, t_cur, y_cur, y_new):
        '''The Jacobian at `t_cur`, given `y_cur` and `y_new`.'''
        # Compute `D`, the derivative of `y_cur` with respect to `y_new`,
        # which is `M_new @ D = M_cur`.
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.birth.rate(t_mid)
        M_new = (self.H
                 - self.t_step / 2 * (self.F
                                      + self.beta @ y_new * self.T
                                      + numpy.outer(self.T @ y_new, self.beta)
                                      + b_mid * self.B))
        M_cur = (self.H
                 + self.t_step / 2 * (self.F
                                      + self.beta @ y_cur * self.T
                                      + numpy.outer(self.T @ y_cur, self.beta)
                                      + b_mid * self.B))
        D = scipy.linalg.solve(M_new, M_cur,
                               overwrite_a=True,
                               overwrite_b=True)
        J = (D - self.I) / self.t_step
        return J


def solve(model, t_span, y_0,
          t=None, y=None, display=False):
    '''Solve the model.'''
    solver = Solver(model)
    return solver.solve(t_span, y_0,
                        t=t, y=y, display=display)
