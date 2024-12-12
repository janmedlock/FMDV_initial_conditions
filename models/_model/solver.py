'''Solver base class.'''

import abc
import functools

import numpy

from . import _jacobian
from .. import _utility


class Solver(metaclass=abc.ABCMeta):
    '''Base class for Crank–Nicolson solvers.'''

    @property
    @abc.abstractmethod
    def _sparse(self):
        '''Whether the solver uses sparse matrices and sparse linear
        algebra.'''

    @property
    @abc.abstractmethod
    def _jacobian_method_default(self):
        '''The default Jacobian method.'''

    def __init__(self, model, _jacobian_method=None, _check_matrices=True):
        self.model = model
        if _jacobian_method is None:
            _jacobian_method = self._jacobian_method_default
        self._jacobian_method = _jacobian_method
        self.t_step = model.t_step
        self._build_matrices()
        if _check_matrices:
            self._check_matrices()
        if self._sparse:
            self._root_kwds = {
                'options': {
                    'jac_options': {
                        'inner_M': self._preconditioner,
                    },
                },
            }
        else:
            self._root_kwds = {}

    @abc.abstractmethod
    def _beta(self):
        '''Build the transmission rate vector beta.'''

    @abc.abstractmethod
    def _I(self):
        '''Build the identity matrix.'''

    @abc.abstractmethod
    def _H(self, q):
        '''Build the time-step matrix H(q).'''

    @abc.abstractmethod
    def _F(self, q):
        '''Build the transition matrix F(q).'''

    @abc.abstractmethod
    def _B(self):
        '''Build the birth matrix B.'''

    @abc.abstractmethod
    def _T(self, q):
        '''Build the transition matrix F(q).'''

    def _A(self, q):
        '''Build the matrix A(q).'''
        H = self._H(q)
        F = self.t_step / 2 * self._F(q)
        if q == 'cur':
            A = H + F
        elif q == 'new':
            A = H - F
        else:
            raise ValueError(f'{q=}')
        return A

    def _build_matrices(self):
        '''Build matrices needed by the solver.'''
        q_vals = ('new', 'cur')
        self.beta = self._beta()
        self.I = self._I()
        self.A = {q: self._A(q) for q in q_vals}
        self.B = self._B()
        self.T = {q: self._T(q) for q in q_vals}

    def _check_matrices(self, is_M_matrix=True):
        '''Check the solver matrices.'''
        assert _utility.linalg.is_nonnegative(self.beta)
        assert _utility.linalg.is_Z_matrix(self.A['new'])
        assert _utility.linalg.is_nonnegative(self.A['cur'])
        assert _utility.linalg.is_Metzler_matrix(self.B)
        assert _utility.linalg.is_nonnegative(self.B)
        assert _utility.linalg.is_Metzler_matrix(self.T['new'])
        birth = self.model.parameters.birth
        if is_M_matrix:
            AB_new = (
                self.A['new']
                - self.t_step / 2 * birth.rate_max * self.B
            )
            assert _utility.linalg.is_M_matrix(AB_new)
        AB_cur = (
            self.A['cur']
            + self.t_step / 2 * birth.rate_min * self.B
        )
        assert _utility.linalg.is_nonnegative(AB_cur)

    @property
    def _preconditioner(self):
        '''For sparse solvers, the Krylov preconditioner.'''
        return self.A['new']

    def _objective(self, y_new, AB_new, ABTy_cur):
        '''Helper for `.step()`.'''
        ABT_new = (
            AB_new
            - self.t_step / 2 * self.beta @ y_new * self.T['new']
        )
        return ABT_new @ y_new - ABTy_cur

    def step(self, t_cur, y_cur, display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.parameters.birth.rate(t_mid)
        AB_new = (
            self.A['new']
            - self.t_step / 2 * b_mid * self.B
        )
        ABT_cur = (
            self.A['cur']
            + self.t_step / 2 * (b_mid * self.B
                                 + self.beta @ y_cur * self.T['cur'])
        )
        ABTy_cur = ABT_cur @ y_cur
        y_new_guess = y_cur
        result = _utility.optimize.root(self._objective, y_new_guess,
                                        args=(AB_new, ABTy_cur),
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
