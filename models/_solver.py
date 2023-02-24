'''Solver base class.'''

import abc

import numpy

from . import _utility


class Base(metaclass=abc.ABCMeta):
    '''Base class for Crank–Nicolson solvers.'''

    def __init__(self, model):
        self.model = model
        self._build_matrices()
        self._check_matrices()

    @property
    @abc.abstractmethod
    def _sparse(self):
        '''Whether the solver uses sparse matrices and sparse linear
        algebra.'''

    @abc.abstractmethod
    def _build_matrices(self):
        '''Build matrices needed by the solver.'''

    def _check_matrices(self):
        '''Check the solver matrices.'''
        assert _utility.is_nonnegative(self.beta)
        assert _utility.is_Z_matrix(self.H_new)
        assert _utility.is_nonnegative(self.H_cur)
        assert _utility.is_Metzler_matrix(self.F_new)
        assert _utility.is_Metzler_matrix(self.T_new)
        assert _utility.is_Metzler_matrix(self.B)
        assert _utility.is_nonnegative(self.B)
        HFB_new = (self.H_new
                   - self.t_step / 2 * (self.F_new
                                        + self.model.birth.rate_max * self.B))
        assert _utility.is_M_matrix(HFB_new)
        HFB_cur = (self.H_cur
                   + self.t_step / 2 * (self.F_cur
                                        + self.model.birth.rate_min * self.B))
        assert _utility.is_nonnegative(HFB_cur)

    @abc.abstractmethod
    def step(self, t_cur, y_cur, display=False):
        '''Do a step.'''

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

    @classmethod
    def _solve_direct(cls, model, t_span, y_0,
                      t=None, y=None, display=False):
        '''Solve the model.'''
        solver = cls(model)
        return solver.solve(t_span, y_0,
                            t=t, y=y, display=display)
