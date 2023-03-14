'''Model base class.'''

import abc
import functools

import pandas

from . import _equilibrium, _limit_cycle
from .. import _utility


class Model(metaclass=abc.ABCMeta):
    '''Base class for models.'''

    @property
    @abc.abstractmethod
    def _Parameters(self):
        '''The parameters class.'''

    @property
    @abc.abstractmethod
    def _Solver(self):
        '''The solver class.'''

    def __init__(self, t_step,
                 _check_matrices=True, _jacobian_method=None,
                 **kwds):
        assert t_step > 0
        self.t_step = t_step
        self._check_matrices = _check_matrices
        self._jacobian_method = _jacobian_method
        self.parameters = self._Parameters(**kwds)
        self._init_post()

    def _init_post(self):
        '''Final initialization.'''
        self._index = self._build_index()

    @functools.cached_property
    def _solver(self):
        '''`._solver` is built on first use and then reused.'''
        _solver = self._Solver(self,
                               _check_matrices=self._check_matrices,
                               _jacobian_method=self._jacobian_method)
        return _solver

    def _get_index_level(self, level):
        '''Get the index for `level`.'''
        if isinstance(self._index, pandas.MultiIndex):
            which = self._index.names.index(level)
            idx_level = self._index.levels[which]
            return idx_level
        else:
            return self._index

    def _build_index(self):
        '''Build a `pandas.Index()` for solutions.'''

    def _build_weights(self):
        '''Build weights for the state vector.'''

    @functools.cached_property
    def _weights(self):
        '''`._weights` is built on first use and then reused.'''
        _weights = self._build_weights()
        return _weights

    def build_initial_conditions(self):
        '''Build the initial conditions.'''

    def Solution(self, y, t=None):
        '''A solution.'''
        if t is None:
            return pandas.Series(y, index=self._index)
        else:
            t = pandas.Index(t, name='time')
            return pandas.DataFrame(y, index=t, columns=self._index)

    def solve(self, t_span,
              y_start=None, t=None, y=None, display=False):
        '''Solve the ODEs.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        (t_, soln) = self._solver.solve(t_span, y_start,
                                        t=t, y=y, display=display)
        _utility.numerical.assert_nonnegative(soln)
        return self.Solution(soln, t_)

    def solution_at_t_end(self, t_span,
                          y_start=None, t=None, y=None, display=False):
        '''Get the solution at the end time.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        y_end = self._solver.solution_at_t_end(t_span, y_start,
                                               t=t, y=y, display=display)
        _utility.numerical.assert_nonnegative(y_end)
        return self.Solution(y_end)

    def find_equilibrium(self, eql_guess, t=0, t_solve=0,
                         display=False, **root_kwds):
        '''Find an equilibrium of the model.'''
        eql = _equilibrium.find(self, eql_guess, t,
                                t_solve=t_solve, weights=self._weights,
                                display=display, **root_kwds)
        _utility.numerical.assert_nonnegative(eql)
        return self.Solution(eql)

    def get_eigenvalues(self, eql, t=0, k=5):
        '''Get the eigenvalues of the Jacobian.'''
        return _equilibrium.eigenvalues(self, eql, t, k=k)

    def find_limit_cycle(self, period_0, t_0, lcy_0_guess,
                         solution=True, display=False, **root_kwds):
        '''Find a limit cycle of the model.'''
        result = _limit_cycle.find_subharmonic(self, period_0, t_0,
                                               lcy_0_guess,
                                               weights=self._weights,
                                               solution=solution,
                                               display=display,
                                               **root_kwds)
        if solution:
            (t, lcy) = result
            _utility.numerical.assert_nonnegative(lcy)
            return self.Solution(lcy, t)
        else:
            lcy_0 = result
            _utility.numerical.assert_nonnegative(lcy_0)
            return self.Solution(lcy_0)

    def get_characteristic_multipliers(self, lcy, k=5, display=False):
        '''Get the characteristic multipliers.'''
        return _limit_cycle.characteristic_multipliers(self, lcy, k=k,
                                                       display=display)

    def get_characteristic_exponents(self, lcy, k=5, display=False):
        '''Get the characteristic exponents.'''
        return _limit_cycle.characteristic_exponents(self, lcy, k=k,
                                                     display=display)
