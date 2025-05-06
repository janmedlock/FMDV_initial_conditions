'''Solver base class.'''

import abc
import functools

import numpy

from . import _crank_nicolson, _jacobian
from .. import _utility


class Population(_crank_nicolson.Mixin, metaclass=abc.ABCMeta):
    '''Base class for Crank–Nicolson solvers for the population and
    infection models.'''

    @property
    @abc.abstractmethod
    def sparse(self):
        '''Whether the solver uses sparse arrays and sparse linear
        algebra.'''

    @functools.cached_property
    @abc.abstractmethod
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''

    @abc.abstractmethod
    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''

    @abc.abstractmethod
    def F(self, q):  # pylint: disable=invalid-name
        '''The transition matrix, F(q).'''

    @functools.cached_property
    @abc.abstractmethod
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''

    def __init__(self, t_step, parameters):
        self.t_step = t_step
        self.parameters = parameters
        super().__init__()

    @staticmethod
    def _integration_vector(n, step):
        '''The integration vector of length `n` and step size `step`.'''
        return step * numpy.ones((1, n))

    def _integration_against_vector(self, n, step, vec):
        '''The vector for integration against `vec` with step size
        `step`.'''
        if numpy.isscalar(vec):
            vec *= numpy.ones(n)
        if self.sparse:
            val = _utility.sparse.Array(step * vec,
                                        shape=(1, n))
        else:
            val = numpy.array(step * vec) \
                       .reshape((1, n))
        return val

    def _influx_vector(self, n, step):
        '''The influx vector of length `n` and step size `step`.'''
        val = _utility.sparse.array_from_dict(
            {(0, 0): 1 / step},
            shape=(n, 1)
        )
        if not self.sparse:
            val = val.todense()
        return val

    def _lag_matrix(self, n):
        '''The lag matrix of shape `n` x `n`.'''
        diags = {
            -1: numpy.ones(n - 1),
            0: numpy.hstack([numpy.zeros(n - 1), 1]),
        }
        val = _utility.sparse.diags_from_dict(diags)
        if not self.sparse:
            val = val.todense()
        return val

    def _A(self, q):  # pylint: disable=invalid-name
        '''The matrix A(q).'''
        return self._cn_op(q,
                           self.H(q),
                           self.F(q))

    @functools.cached_property
    def A(self):  # pylint: disable=invalid-name
        '''The matrix A(q).'''
        # Lazily evaluated
        # `{q: self._A(q)
        #   for q in self._q_vals}`
        return _utility.lazy.Dict({
            q: (self._A, (q, ))
            for q in self._q_vals
        })

    @functools.cached_property
    def A_hat_new(self):  # pylint: disable=invalid-name
        r'''The matrix \hat{A}_{new}.'''
        return self.t_step / 2 * self.F('new')

    # pylint: disable-next=invalid-name
    def _check_matrices(self, is_M_matrix=True):
        '''Check the solver matrices.'''
        assert _utility.linalg.is_Z_matrix(self.A['new'])
        assert _utility.linalg.is_nonnegative(self.A['cur'])
        assert _utility.linalg.is_Metzler_matrix(self.B)
        assert _utility.linalg.is_nonnegative(self.B)
        if is_M_matrix:
            # pylint: disable-next=invalid-name
            A_B_new = self._cn_op('new',
                                  self.A['new'],
                                  self.parameters.birth.rate_max * self.B)
            assert _utility.linalg.is_M_matrix(A_B_new)
        # pylint: disable-next=invalid-name
        A_B_cur = self._cn_op('cur',
                              self.A['cur'],
                              self.parameters.birth.rate_min * self.B)
        assert _utility.linalg.is_nonnegative(A_B_cur)


class Solver(Population, metaclass=abc.ABCMeta):
    '''Base class for Crank–Nicolson solvers for the infection models.'''

    @property
    @abc.abstractmethod
    def _jacobian_method_default(self):
        '''The default Jacobian method.'''

    @functools.cached_property
    @abc.abstractmethod
    def beta(self):
        '''The transmission rate vector.'''

    @abc.abstractmethod
    def _T(self, q):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''

    def __init__(self, model, _jacobian_method=None, _check_matrices=True):
        super().__init__(model.t_step, model.parameters)
        self.model = model
        if _jacobian_method is None:
            _jacobian_method = self._jacobian_method_default
        self._jacobian_method = _jacobian_method
        if _check_matrices:
            self._check_matrices()
        if self.sparse:
            self._root_kwds = {
                'options': {
                    'jac_options': {
                        'inner_M': self._preconditioner,
                    },
                },
            }
        else:
            self._root_kwds = {}
        self._fixed_point_kwds = {}

    @functools.cached_property
    def T(self):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        # Lazily evaluated
        # `{q: self._T(q)
        #   for q in self._q_vals}`
        return _utility.lazy.Dict({
            q: (self._T, (q, ))
            for q in self._q_vals
        })

    # pylint: disable-next=invalid-name
    def _check_matrices(self, is_M_matrix=True):
        '''Check the solver matrices.'''
        super()._check_matrices(is_M_matrix=is_M_matrix)
        assert _utility.linalg.is_nonnegative(self.beta)
        assert _utility.linalg.is_Metzler_matrix(self.T['new'])

    @property
    def _preconditioner(self):
        '''For sparse solvers, the Krylov preconditioner.'''
        return self.A['new']

    # pylint: disable-next=invalid-name
    def _root_objective(self, y_new, A_B_new, A_B_T_y_cur):
        '''Helper for `.step(..., solver='root', ...)`.'''
        # pylint: disable-next=invalid-name
        A_B_T_new = self._cn_op('new',
                                A_B_new,
                                self.beta @ y_new * self.T['new'])
        return A_B_T_new @ y_new - A_B_T_y_cur

    # pylint: disable-next=invalid-name
    def _fixed_point_objective(self, y_new, A_hat_B_new, A_B_T_y_cur):
        '''Helper for `.step(..., solver='fixed_point', ...)`.'''
        # pylint: disable-next=invalid-name
        A_hat_B_T_new = self._cn_op('cur',
                                    A_hat_B_new,
                                    self.beta @ y_new * self.T['new'])
        return A_hat_B_T_new @ y_new + A_B_T_y_cur

    def step(self, t_cur, y_cur, solver='root', display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.parameters.birth.rate(t_mid)
        # pylint: disable-next=invalid-name
        A_B_T_cur = self._cn_op(
            'cur',
            self.A['cur'],
            b_mid * self.B + self.beta @ y_cur * self.T['cur']
        )
        # pylint: disable-next=invalid-name
        A_B_T_y_cur = A_B_T_cur @ y_cur
        y_new_guess = y_cur
        if solver == 'root':
            # pylint: disable-next=invalid-name
            A_B_new = self._cn_op('new',
                                  self.A['new'],
                                  b_mid * self.B)
            y_new = _utility.optimize.root(self._root_objective, y_new_guess,
                                           args=(A_B_new, A_B_T_y_cur),
                                           sparse=self.sparse,
                                           **self._root_kwds)
        elif solver == 'fixed_point':
            # pylint: disable-next=invalid-name
            A_hat_B_new = self._cn_op('cur',
                                      self.A_hat_new,
                                      b_mid * self.B)
            try:
                y_new = _utility.optimize.fixed_point(
                    self._fixed_point_objective, y_new_guess,
                    args=(A_hat_B_new, A_B_T_y_cur),
                    **self._fixed_point_kwds
                )
            except RuntimeError as err:
                raise RuntimeError(f'{t_cur=:g}') from err
        else:
            raise ValueError(f'Unknown {solver=}!')
        return y_new

    # pylint: disable-next=too-many-arguments
    def solve(self, t_span, y_0, *,
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

    # pylint: disable-next=too-many-arguments
    def solution_at_t_end(self, t_span, y_0, *,
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
        return _jacobian.Jacobian(self, method=self._jacobian_method)

    def jacobian(self, t_cur, y_cur, y_new):
        '''Calculate the Jacobian at `t_cur`, given `y_cur` and `y_new`.'''
        return self._jacobian(t_cur, y_cur, y_new)
