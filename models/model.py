'''Model base class.'''

import numpy

from . import birth
from . import death
from . import parameters
from . import progression
from . import recovery
from . import transmission
from . import waning


class Base:
    '''Base class for models.'''

    states = ('maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered')

    # This determines whether offspring are born with maternal
    # immunity.
    states_with_antibodies = ['recovered']

    # For easy indexing, whether each state has antibodies.
    _states_have_antibodies = numpy.isin(states,
                                         states_with_antibodies)

    def __init__(self, **kwds):
        parameters_ = parameters.Parameters(**kwds)
        self.death = death.Death(parameters_)
        self.birth = birth.Birth(parameters_, self.death)
        self.progression = progression.Progression(parameters_)
        self.recovery = recovery.Recovery(parameters_)
        self.transmission = transmission.Transmission(parameters_)
        self.waning = waning.Waning(parameters_)


class AgeIndependent(Base):
    '''Base class for age-independent models.'''

    def __init__(self, **kwds):
        super().__init__(**kwds)
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
