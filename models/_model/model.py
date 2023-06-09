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

    # The default time step `t_step`. A necessary condition for
    # nonnegative solutions is is that `t_step` must be less than 1 /
    # rate for all of the model transition rates. In particular,
    # `transmission_rate` and 1 / `progression_mean`, especially for
    # SAT1, are just a bit less than 1000.
    _t_step_default = 1e-3

    def __init__(self,
                 t_step=_t_step_default,
                 _solver_options=None,
                 **parameters_kwds):
        assert t_step > 0
        self.t_step = t_step
        if _solver_options is None:
            _solver_options = {}
        self._solver_options = _solver_options
        self.parameters = self._Parameters(**parameters_kwds)
        self._index = self._build_index()
        super().__init__()

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
        assert not hasattr(super(), '_build_index')
        return None

    @functools.cached_property
    def _weights(self):
        '''Weights for the state vector.'''
        assert not hasattr(super(), '_weights')
        return None

    def build_initial_conditions(self):
        '''Build the initial conditions.'''
        assert not hasattr(super(), 'build_initial_conditions')
        return None

    @functools.cached_property
    def _solver(self):
        '''`._solver` is built on first use and then reused.'''
        _solver = self._Solver(self, **self._solver_options)
        return _solver

    def Solution(self, y, t=None):
        '''A solution.'''
        if t is None:
            return pandas.Series(y, index=self._index)
        else:
            t = pandas.Index(t, name='time')
            return pandas.DataFrame(y, index=t, columns=self._index)

    def solve(self, t_span,
              y_start=None, t=None, y=None, display=False,
              check_nonnegative=True):
        '''Solve the ODEs.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        (t_, soln_) = self._solver.solve(t_span, y_start,
                                        t=t, y=y, display=display)
        soln = self.Solution(soln_, t_)
        if check_nonnegative:
            _utility.numerical.assert_nonnegative(soln)
        return soln

    def solution_at_t_end(self, t_span,
                          y_start=None, t=None, y_temp=None,
                          display=False, check_nonnegative=True):
        '''Get the solution at the end time.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        y_end = self._solver.solution_at_t_end(t_span, y_start,
                                               t=t, y_temp=y_temp,
                                               display=display)
        soln = self.Solution(y_end)
        if check_nonnegative:
            _utility.numerical.assert_nonnegative(soln)
        return soln

    def find_equilibrium(self, eql_guess, t=0, t_solve=0,
                         display=False, check_nonnegative=True,
                         **root_kwds):
        '''Find an equilibrium of the model.'''
        eql_ = _equilibrium.find(self, eql_guess, t,
                                 t_solve=t_solve, weights=self._weights,
                                 display=display, **root_kwds)
        eql = self.Solution(eql_)
        if check_nonnegative:
            _utility.numerical.assert_nonnegative(eql)
        return eql

    def get_eigenvalues(self, eql, t=0, k=5):
        '''Get the eigenvalues of the Jacobian.'''
        return _equilibrium.eigenvalues(self, eql, t, k=k)

    def find_limit_cycle(self, period_0, t_0, lcy_0_guess,
                         solution=True, display=False,
                         check_nonnegative=True,
                         **root_kwds):
        '''Find a limit cycle of the model.'''
        result = _limit_cycle.find_subharmonic(self, period_0, t_0,
                                               lcy_0_guess,
                                               weights=self._weights,
                                               solution=solution,
                                               display=display,
                                               **root_kwds)
        if solution:
            (t, lcy) = result
            soln = self.Solution(lcy, t)
        else:
            lcy_0 = result
            soln = self.Solution(lcy_0)
        if check_nonnegative:
            _utility.numerical.assert_nonnegative(soln)
        return soln

    def get_characteristic_multipliers(self, lcy, k=5, display=False):
        '''Get the characteristic multipliers.'''
        return _limit_cycle.characteristic_multipliers(self, lcy, k=k,
                                                       display=display)

    def get_characteristic_exponents(self, lcy, k=5, display=False):
        '''Get the characteristic exponents.'''
        return _limit_cycle.characteristic_exponents(self, lcy, k=k,
                                                     display=display)
