'''Model base class.'''

import abc
import functools

import pandas

from . import _equilibrium, _limit_cycle
from .. import _utility


# pylint: disable-next=too-few-public-methods
class Population(metaclass=abc.ABCMeta):
    '''Base class for the population and infection models.'''

    @property
    @abc.abstractmethod
    def _Solver(self):  # pylint: disable=invalid-name
        '''The solver class.'''

    def __init__(self, t_step, parameters, **solver_kwds):
        assert t_step > 0
        self.t_step = t_step
        self.parameters = parameters
        self.solver_kwds = solver_kwds
        super().__init__()

    @functools.cached_property
    def solver(self):
        '''`.solver` is built on first use and then reused.'''
        return self._Solver(self, **self.solver_kwds)


class Model(Population, metaclass=abc.ABCMeta):
    '''Base class for the infection models.'''

    @property
    @abc.abstractmethod
    def _Parameters(self):  # pylint: disable=invalid-name
        '''The parameters class.'''

    # The default time step `t_step`. A necessary condition for
    # nonnegative solutions is is that `t_step` must be less than 1 /
    # rate for all of the model transition rates. In particular,
    # `transmission_rate` and 1 / `progression_mean`, especially for
    # SAT1, are just a bit less than 1000.
    _t_step_default = 1e-3

    def __init__(self,
                 t_step=_t_step_default,
                 solver_kwds=None,
                 **parameters_kwds):
        if solver_kwds is None:
            solver_kwds = {}
        super().__init__(t_step,
                         self._Parameters(**parameters_kwds),
                         **solver_kwds)
        # pylint: disable-next=assignment-from-none
        self._index = self._build_index()

    def _get_index_level(self, level):
        '''Get the index for `level`.'''
        if isinstance(self._index, pandas.MultiIndex):
            which = self._index.names.index(level)
            idx_level = self._index.levels[which]
        else:
            idx_level = self._index
        return idx_level

    def _build_index(self):  # pylint: disable=useless-return
        '''Build a `pandas.Index()` for solutions.'''
        assert not hasattr(super(), '_build_index')
        return None

    @functools.cached_property
    def _weights(self):  # pylint: disable=useless-return
        '''Weights for the state vector.'''
        assert not hasattr(super(), '_weights')
        return None

    def build_initial_conditions(self):  # pylint: disable=useless-return
        '''Build the initial conditions.'''
        assert not hasattr(super(), 'build_initial_conditions')
        return None

    def Solution(self, y, t=None):  # pylint: disable=invalid-name
        '''A solution.'''
        if t is None:
            solution = pandas.Series(y, index=self._index)
        else:
            t = pandas.Index(t, name='time')
            solution = pandas.DataFrame(y, index=t, columns=self._index)
        return solution

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def solve(self, t_span,
              y_start=None, t=None, y=None, display=False,
              check_nonnegative=True):
        '''Solve the ODEs.'''
        if y_start is None:
            # pylint: disable-next=assignment-from-none
            y_start = self.build_initial_conditions()
        (t_, soln_) = self.solver.solve(t_span, y_start,
                                        t=t, y=y, display=display)
        soln = self.Solution(soln_, t_)
        if check_nonnegative:
            _utility.numerical.check_nonnegative(soln)
        return soln

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def solution_at_t_end(self, t_span,
                          y_start=None, t=None, y_temp=None,
                          display=False, check_nonnegative=True):
        '''Get the solution at the end time.'''
        if y_start is None:
            # pylint: disable-next=assignment-from-none
            y_start = self.build_initial_conditions()
        y_end = self.solver.solution_at_t_end(t_span, y_start,
                                              t=t, y_temp=y_temp,
                                              display=display)
        soln = self.Solution(y_end)
        if check_nonnegative:
            _utility.numerical.check_nonnegative(soln)
        return soln

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def find_equilibrium(self, eql_guess, t=0, t_solve=0,
                         display=False, check_nonnegative=True,
                         **root_kwds):
        '''Find an equilibrium of the model.'''
        eql_ = _equilibrium.find(self, eql_guess, t,
                                 t_solve=t_solve, weights=self._weights,
                                 display=display, **root_kwds)
        eql = self.Solution(eql_)
        if check_nonnegative:
            _utility.numerical.check_nonnegative(eql)
        return eql

    def get_eigenvalues(self, eql, t=0, k=5, **kws):
        '''Get the eigenvalues of the Jacobian.'''
        return _equilibrium.eigenvalues(self, eql, t, k=k, **kws)

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
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
            _utility.numerical.check_nonnegative(soln)
        return soln

    def check_limit_cycle(self, sol, **kwds):
        '''Check whether `sol` is a limit cycle.'''
        y = sol.to_numpy()
        return _limit_cycle.check(y,
                                  weights=self._weights,
                                  **kwds)

    def get_characteristic_multipliers(self, lcy, k=5, display=False):
        '''Get the characteristic multipliers.'''
        return _limit_cycle.characteristic_multipliers(self, lcy, k=k,
                                                       display=display)

    def get_characteristic_exponents(self, lcy, k=5, display=False):
        '''Get the characteristic exponents.'''
        return _limit_cycle.characteristic_exponents(self, lcy, k=k,
                                                     display=display)
