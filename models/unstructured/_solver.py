'''Solver.'''

import numpy
import scipy.linalg
import scipy.optimize

from .. import _solver
from .. import _utility


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = False

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
        self.H_new = self.H_cur = self._H()
        self.F_new = self.F_cur = self._F()
        self.T_new = self.T_cur = self._T()
        self.B = self._B()

    def _objective(self, y_new, HFB_new, HFTBy_cur):
        '''Helper for `.step()`.'''
        HFTB_new = (HFB_new
                    - self.t_step / 2 * self.beta @ y_new * self.T_new)
        return HFTB_new @ y_new - HFTBy_cur

    def step(self, t_cur, y_cur, display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.birth.rate(t_mid)
        HFB_new = (self.H_new
                   - self.t_step / 2 * (self.F_new
                                        + b_mid * self.B))
        HFTB_cur = (self.H_cur
                    + self.t_step / 2 * (self.F_cur
                                         + self.beta @ y_cur * self.T_cur
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
        M_new = (self.H_new
                 - self.t_step / 2 * (self.F_new
                                      + self.beta @ y_new * self.T_new
                                      + numpy.outer(self.T_new @ y_new,
                                                    self.beta)
                                      + b_mid * self.B))
        M_cur = (self.H_cur
                 + self.t_step / 2 * (self.F_cur
                                      + self.beta @ y_cur * self.T_cur
                                      + numpy.outer(self.T_cur @ y_cur,
                                                    self.beta)
                                      + b_mid * self.B))
        D = scipy.linalg.solve(M_new, M_cur,
                               overwrite_a=True,
                               overwrite_b=True)
        J = (D - self.I) / self.t_step
        return J


solve = Solver._solve_direct
