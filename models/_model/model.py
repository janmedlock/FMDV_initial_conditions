'''Model base class.'''

import abc

import pandas

from . import _equilibrium, _limit_cycle
from .. import _utility


class Model(metaclass=abc.ABCMeta):
    '''Base class for models.'''

    @property
    @abc.abstractmethod
    def _Solver(self):
        '''The solver class.'''

    @abc.abstractmethod
    def _init_parameters(self, **kwds):
        '''Initialize model parameters.'''

    def __init__(self, t_step, **kwds):
        self.t_step = t_step
        self._init_parameters(**kwds)
        self._solver = self._Solver(self, t_step)
        self._index = self._build_index()
        self._weights = self._build_weights()

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

    def find_equilibrium(self, eql_guess, t=0, **root_kwds):
        '''Find an equilibrium of the model.'''
        eql = _equilibrium.find(self, eql_guess, t,
                                weights=self._weights, **root_kwds)
        _utility.numerical.assert_nonnegative(eql)
        return self.Solution(eql)

    def get_eigenvalues(self, eql, t=0, k=5):
        '''Get the eigenvalues of the Jacobian.'''
        return _equilibrium.eigenvalues(self, eql, t, k=k)

    def find_limit_cycle(self, period_0, t_0, lcy_0_guess,
                         solution=True, **root_kwds):
        '''Find a limit cycle of the model.'''
        result = _limit_cycle.find_subharmonic(self, period_0, t_0,
                                               lcy_0_guess,
                                               weights=self._weights,
                                               solution=solution,
                                               **root_kwds)
        if solution:
            (t, lcy) = result
            _utility.numerical.assert_nonnegative(lcy)
            return self.Solution(lcy, t)
        else:
            lcy_0 = result
            _utility.numerical.assert_nonnegative(lcy_0)
            return self.Solution(lcy_0)

    def get_characteristic_multipliers(self, lcy, k=5):
        '''Get the characteristic multipliers.'''
        return _limit_cycle.characteristic_multipliers(self, lcy, k=k)

    def get_characteristic_exponents(self, lcy, k=5):
        '''Get the characteristic exponents.'''
        return _limit_cycle.characteristic_exponents(self, lcy, k=k)
