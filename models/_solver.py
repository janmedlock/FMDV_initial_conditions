'''Solver base class.'''

import abc

import numpy

from . import _utility
from ._utility import linalg
from ._utility import optimize


class Base(metaclass=abc.ABCMeta):
    '''Base class for Crank–Nicolson solvers.'''

    def __init__(self, model):
        self.model = model
        self._build_matrices()
        self._check_matrices()
        if self._sparse:
            self._root_kwds = dict(
                options=dict(jac_options=dict(inner_M=self._preconditioner()))
            )
        else:
            self._root_kwds = dict()

    @property
    @abc.abstractmethod
    def _sparse(self):
        '''Whether the solver uses sparse matrices and sparse linear
        algebra.'''

    @abc.abstractmethod
    def _I(self): pass

    @abc.abstractmethod
    def _beta(self): pass

    @abc.abstractmethod
    def _Hq(self, q): pass

    @abc.abstractmethod
    def _Fq(self, q): pass

    @abc.abstractmethod
    def _Tq(self, q): pass

    @abc.abstractmethod
    def _B(self): pass

    def _build_matrices(self):
        '''Build matrices needed by the solver.'''
        self.I = self._I()
        self.beta = self._beta()
        self.H_new = self._Hq('new')
        self.H_cur = self._Hq('cur')
        self.F_new = self._Fq('new')
        self.F_cur = self._Fq('cur')
        self.T_new = self._Tq('new')
        self.T_cur = self._Tq('cur')
        self.B = self._B()

    def _check_matrices(self):
        '''Check the solver matrices.'''
        assert linalg.is_nonnegative(self.beta)
        assert linalg.is_Z_matrix(self.H_new)
        assert linalg.is_nonnegative(self.H_cur)
        assert linalg.is_Metzler_matrix(self.F_new)
        assert linalg.is_Metzler_matrix(self.T_new)
        assert linalg.is_Metzler_matrix(self.B)
        assert linalg.is_nonnegative(self.B)
        HFB_new = (self.H_new
                   - self.t_step / 2 * (self.F_new
                                        + self.model.birth.rate_max * self.B))
        assert linalg.is_M_matrix(HFB_new)
        HFB_cur = (self.H_cur
                   + self.t_step / 2 * (self.F_cur
                                        + self.model.birth.rate_min * self.B))
        assert linalg.is_nonnegative(HFB_cur)

    def _preconditioner(self):
        M = (self.H_new
             + self.t_step / 2 * self.F_new)
        return M

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
        result = optimize.root(self._objective, y_new_guess,
                               args=(HFB_new, HFTBy_cur),
                               sparse=self._sparse,
                               **self._root_kwds)
        assert result.success, f'{t_cur=}\n{result=}'
        y_new = result.x
        return y_new

    def solve(self, t_span, y_0,
              t=None, y=None, display=False):
        '''Solve. `y` is storage for the solution, which will be built
        if not provided.'''
        if t is None:
            t = _utility.build_t(*t_span, self.t_step)
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
        y[0] = y_0
        for ell in range(1, len(t)):
            y[ell] = self.step(t[ell - 1], y[ell - 1], display=display)
        return (t, y)

    def solution_at_t_end(self, t_span, y_0,
                          t=None, y_temp=None):
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
        D = linalg.solve(M_new, M_cur,
                         overwrite_a=True,
                         overwrite_b=True)
        J = (D - self.I) / self.t_step
        return J
