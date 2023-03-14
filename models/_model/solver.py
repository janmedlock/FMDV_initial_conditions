'''Solver base class.'''

import abc
import functools

import numpy

from . import _jacobian
from .. import _utility


class Base(metaclass=abc.ABCMeta):
    '''Base class for Crank–Nicolson solvers.'''

    @property
    @abc.abstractmethod
    def _sparse(self):
        '''Whether the solver uses sparse matrices and sparse linear
        algebra.'''

    def __init__(self, model, _check_matrices=True, _jacobian_method=None):
        self.model = model
        self._jacobian_method = _jacobian_method
        self.t_step = model.t_step
        self._build_matrices()
        if _check_matrices:
            self._check_matrices()
        if self._sparse:
            self._root_kwds = dict(
                options=dict(jac_options=dict(inner_M=self._preconditioner()))
            )
        else:
            self._root_kwds = dict()

    @abc.abstractmethod
    def _I(self):
        '''Build the identity matrix.'''

    @abc.abstractmethod
    def _beta(self):
        '''Build the transmission rate vector beta.'''

    @abc.abstractmethod
    def _H(self, q):
        '''Build the time step matrix H(q).'''

    @abc.abstractmethod
    def _F(self, q):
        '''Build the transition matrix F(q).'''

    @abc.abstractmethod
    def _T(self, q):
        '''Build the transition matrix F(q).'''

    @abc.abstractmethod
    def _B(self):
        '''Build the birth matrix B.'''

    def _build_matrices(self):
        '''Build matrices needed by the solver.'''
        self.I = self._I()
        self.beta = self._beta()
        q_vals = ('new', 'cur')
        self.H = {q: self._H(q) for q in q_vals}
        self.F = {q: self._F(q) for q in q_vals}
        self.T = {q: self._T(q) for q in q_vals}
        self.B = self._B()

    def _check_matrices(self):
        '''Check the solver matrices.'''
        assert _utility.linalg.is_nonnegative(self.beta)
        assert _utility.linalg.is_Z_matrix(self.H['new'])
        assert _utility.linalg.is_nonnegative(self.H['cur'])
        assert _utility.linalg.is_Metzler_matrix(self.F['new'])
        assert _utility.linalg.is_Metzler_matrix(self.T['new'])
        assert _utility.linalg.is_Metzler_matrix(self.B)
        assert _utility.linalg.is_nonnegative(self.B)
        birth = self.model.parameters.birth
        HFB_new = (self.H['new']
                   - self.t_step / 2 * (self.F['new']
                                        + birth.rate_max * self.B))
        assert _utility.linalg.is_M_matrix(HFB_new)
        HFB_cur = (self.H['cur']
                   + self.t_step / 2 * (self.F['cur']
                                        + birth.rate_min * self.B))
        assert _utility.linalg.is_nonnegative(HFB_cur)

    def _preconditioner(self):
        '''For sparse solvers, build the Krylov preconditioner.'''
        M = (self.H['new']
             + self.t_step / 2 * self.F['new'])
        return M

    def _objective(self, y_new, HFB_new, HFTBy_cur):
        '''Helper for `.step()`.'''
        HFTB_new = (HFB_new
                    - self.t_step / 2 * self.beta @ y_new * self.T['new'])
        return HFTB_new @ y_new - HFTBy_cur

    def step(self, t_cur, y_cur, display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.parameters.birth.rate(t_mid)
        HFB_new = (self.H['new']
                   - self.t_step / 2 * (self.F['new']
                                        + b_mid * self.B))
        HFTB_cur = (self.H['cur']
                    + self.t_step / 2 * (self.F['cur']
                                         + self.beta @ y_cur * self.T['cur']
                                         + b_mid * self.B))
        HFTBy_cur = HFTB_cur @ y_cur
        y_new_guess = y_cur
        result = _utility.optimize.root(self._objective, y_new_guess,
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
            t = _utility.numerical.build_t(*t_span, self.t_step)
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
        y[0] = y_0
        for ell in range(1, len(t)):
            y[ell] = self.step(t[ell - 1], y[ell - 1], display=display)
        return (t, y)

    def solution_at_t_end(self, t_span, y_0,
                          t=None, y_temp=None, display=False):
        '''Find the value of the solution at `t_span[1]`.'''
        if t is None:
            t = _utility.numerical.build_t(*t_span, self.t_step)
        if y_temp is None:
            y_temp = numpy.empty((2, *numpy.shape(y_0)))
        (y_cur, y_new) = y_temp
        y_new[:] = y_0
        for t_cur in t[:-1]:
            # Update so that what was the new value of the solution is
            # now the current value and what was the current value of
            # the solution will be storage space for the new value.
            (y_cur, y_new) = (y_new, y_cur)
            y_new[:] = self.step(t_cur, y_cur, display=display)
        return y_new

    @functools.cached_property
    def _jacobian(self):
        '''`._jacobian` is built on first use and then reused.'''
        _jacobian_ = _jacobian.Calculator(self, method=self._jacobian_method)
        return _jacobian_

    def jacobian(self, t_cur, y_cur, y_new):
        '''Calculate the Jacobian at `t_cur`, given `y_cur` and `y_new`.'''
        J = self._jacobian.calculate(t_cur, y_cur, y_new)
        return J
