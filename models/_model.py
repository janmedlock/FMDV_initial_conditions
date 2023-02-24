'''Model base class.'''

import abc

import numpy
import pandas

from . import birth
from . import death
from . import parameters
from . import progression
from . import recovery
from . import transmission
from . import waning
from . import _equilibrium
from . import _limit_cycle
from . import _population
from . import _utility


class _Base(metaclass=abc.ABCMeta):
    '''Base class for models.'''

    states = ('maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered')

    # This determines whether offspring are born with maternal
    # immunity.
    states_with_antibodies = ['recovered']

    # For easy indexing, whether each state has antibodies.
    _states_have_antibodies = numpy.isin(states,
                                         states_with_antibodies)

    @property
    @abc.abstractmethod
    def _Solver(self):
        '''A solver instance.'''

    @abc.abstractmethod
    def _build_solution_index(self, states):
        '''Build the solution index.'''

    @abc.abstractmethod
    def _build_weights(self):
        '''Build weights for the state vector.'''

    @abc.abstractmethod
    def build_initial_conditions(self):
        '''Build the initial conditions.'''

    def __init__(self, **kwds):
        self._init_parameters(**kwds)
        self._solver = self._Solver(self)
        states = pandas.CategoricalIndex(self.states, self.states,
                                         ordered=True, name='state')
        self._solution_index = self._build_solution_index(states)
        self._weights = self._build_weights()

    def _init_parameters(self, **kwds):
        parameters_ = parameters.Parameters(**kwds)
        self.death = death.Death(parameters_)
        self.birth = birth.Birth(parameters_, self.death)
        self.progression = progression.Progression(parameters_)
        self.recovery = recovery.Recovery(parameters_)
        self.transmission = transmission.Transmission(parameters_)
        self.waning = waning.Waning(parameters_)

    def Solution(self, y, t=None):
        '''A solution.'''
        if t is None:
            return pandas.Series(y, index=self._solution_index)
        else:
            t = pandas.Index(t, name='time')
            return pandas.DataFrame(y, index=t, columns=self._solution_index)

    def solve(self, t_span,
              y_start=None, t=None, y=None, display=False):
        '''Solve the ODEs.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        (t_, soln) = self._solver.solve(t_span, y_start,
                                        t=t, y=y, display=display)
        _utility.assert_nonnegative(soln)
        return self.Solution(soln, t_)

    def find_equilibrium(self, eql_guess, t=0, **root_kwds):
        '''Find an equilibrium of the model.'''
        eql = _equilibrium.find(self, eql_guess, t,
                                weights=self._weights, **root_kwds)
        _utility.assert_nonnegative(eql)
        return self.Solution(eql)

    def get_eigenvalues(self, eql, t=0, k=5):
        '''Get the eigenvalues of the Jacobian.'''
        return _equilibrium.eigenvalues(self, eql, t, k=k)

    def find_limit_cycle(self, period_0, t_0, lcy_0_guess, **root_kwds):
        '''Find a limit cycle of the model.'''
        (t, lcy) = _limit_cycle.find_subharmonic(self, period_0, t_0,
                                                 lcy_0_guess,
                                                 weights=self._weights,
                                                 **root_kwds)
        _utility.assert_nonnegative(lcy)
        return self.Solution(lcy, t)

    def get_characteristic_multipliers(self, lcy, k=5):
        '''Get the characteristic multipliers.'''
        return _limit_cycle.characteristic_multipliers(self, lcy, k=k)

    def get_characteristic_exponents(self, lcy, k=5):
        '''Get the characteristic exponents.'''
        return _limit_cycle.characteristic_exponents(self, lcy, k=k)


class AgeIndependent(_Base):
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


class AgeDependent(_Base):
    '''Base class for age-dependent models.'''

    def stable_age_density(self, *args, **kwds):
        '''Get the stable age density.'''
        (a, v_dom) = _population.stable_age_density(self.birth, self.death,
                                                    *args, **kwds)
        # Interpolate the logarithm of `v_dom` to `self.a`.
        assert numpy.all(v_dom > 0)
        logn = numpy.interp(self.a, a, numpy.log(v_dom))
        n = numpy.exp(logn)
        # Normalize to integrate to 1.
        n /= n.sum() * self.a_step
        return n
