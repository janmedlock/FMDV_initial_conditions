'''Model base class.'''

import abc

import numpy
import pandas

from . import equilibrium, limit_cycle
from .. import (birth, death, progression, parameters,
                recovery, transmission, waning)
from .._utility import numerical


class Model(metaclass=abc.ABCMeta):
    '''Base class for models.'''

    states = ('maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered')

    # This determines whether offspring are born with maternal
    # immunity.
    states_with_antibodies = ['recovered']

    # For easy indexing, whether each state has antibodies.
    _states_have_antibodies = numpy.isin(states,
                                         states_with_antibodies)

    # The default time step `t_step` for `Model()`. A necessary
    # condition for nonnegative solutions is is that `t_step` must be
    # less than 1 / rate for all of the model transition rates. In
    # particular, `transmission_rate` and 1 / `progression_mean`,
    # especially for SAT1, are just a bit less than 1000.
    DEFAULT_T_STEP = 1e-3

    @property
    @abc.abstractmethod
    def _Solver(self):
        '''The solver class.'''

    @abc.abstractmethod
    def _build_index(self):
        '''Build a `pandas.Index()` for solutions.'''

    @abc.abstractmethod
    def _build_weights(self):
        '''Build weights for the state vector.'''

    @abc.abstractmethod
    def build_initial_conditions(self):
        '''Build the initial conditions.'''

    def __init__(self, t_step=DEFAULT_T_STEP, **kwds):
        self.t_step = t_step
        self._init_parameters(**kwds)
        self._solver = self._Solver(self, t_step)
        self._index = self._build_index()
        self._weights = self._build_weights()

    def _init_parameters(self, **kwds):
        parameters_ = parameters.Parameters(**kwds)
        self.death = death.Death(parameters_)
        self.birth = birth.Birth(parameters_, self.death)
        self.progression = progression.Progression(parameters_)
        self.recovery = recovery.Recovery(parameters_)
        self.transmission = transmission.Transmission(parameters_)
        self.waning = waning.Waning(parameters_)

    def _get_index_level(self, level):
        '''Get the index for `level`.'''
        if isinstance(self._index, pandas.MultiIndex):
            which = self._index.names.index(level)
            idx_level = self._index.levels[which]
            return idx_level
        else:
            return self._index

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
        numerical.assert_nonnegative(soln)
        return self.Solution(soln, t_)

    def find_equilibrium(self, eql_guess, t=0, **root_kwds):
        '''Find an equilibrium of the model.'''
        eql = equilibrium.find(self, eql_guess, t,
                               weights=self._weights, **root_kwds)
        numerical.assert_nonnegative(eql)
        return self.Solution(eql)

    def get_eigenvalues(self, eql, t=0, k=5):
        '''Get the eigenvalues of the Jacobian.'''
        return equilibrium.eigenvalues(self, eql, t, k=k)

    def find_limit_cycle(self, period_0, t_0, lcy_0_guess,
                         solution=True, **root_kwds):
        '''Find a limit cycle of the model.'''
        result = limit_cycle.find_subharmonic(self, period_0, t_0,
                                              lcy_0_guess,
                                              weights=self._weights,
                                              solution=solution,
                                              **root_kwds)
        if solution:
            (t, lcy) = result
            numerical.assert_nonnegative(lcy)
            return self.Solution(lcy, t)
        else:
            lcy_0 = result
            numerical.assert_nonnegative(lcy_0)
            return self.Solution(lcy_0)

    def get_characteristic_multipliers(self, lcy, k=5):
        '''Get the characteristic multipliers.'''
        return limit_cycle.characteristic_multipliers(self, lcy, k=k)

    def get_characteristic_exponents(self, lcy, k=5):
        '''Get the characteristic exponents.'''
        return limit_cycle.characteristic_exponents(self, lcy, k=k)


class ModelAgeIndependent(Model):
    '''Base class for age-independent models.'''

    def _init_parameters(self, **kwds):
        super()._init_parameters(**kwds)
        # Use `self.birth` with age-dependent `.mean` to find
        # `self.death_rate_mean`.
        self.death_rate_mean = self.death.rate_population_mean(self.birth)
        # Set `self.birth.mean` so this age-independent model has
        # zero population growth rate.
        self.birth.mean = self._birth_rate_mean_for_zero_population_growth()

    def _birth_rate_mean_for_zero_population_growth(self):
        '''For this unstructured model, the mean population growth
        rate is `self.birth_rate.mean - self.death_rate_mean`.'''
        return self.death_rate_mean
