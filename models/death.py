'''Death.'''

import numpy
import scipy.integrate

from . import _population


class DeathRate:
    '''Death rate.'''

    # Annual survival in age bands (ages in years).
    # The bands are closed on the left, open on the right.
    _annual_survival = {(0, 1): 0.66,
                        (1, 3): 0.79,
                        (3, 12): 0.88,
                        (12, numpy.inf): 0.66}

    # Death rate values with units per year.
    _death_rate = {key: -numpy.log(val)
                   for (key, val)
                   in _annual_survival.items()}

    def __init__(self, parameters):
        # The death rate does not depend on `parameters`.
        pass

    def __call__(self, age):
        '''Death rate.'''
        # `condlist` is a list of arrays of whether `age` is in each
        # interval in `self._death_rate`.
        condlist = [(left <= age) & (age < right)
                    for (left, right) in self._death_rate.keys()]
        return numpy.select(condlist, self._death_rate.values(), numpy.nan)

    def population_mean(self, birth_rate, maternity_rate,
                        *args, **kwds):
        '''Get the mean death rate when the population is at the stable
        age density.'''
        (ages, density) = _population.stable_age_density(birth_rate, self,
                                                         maternity_rate,
                                                         *args, **kwds)
        rate_total = scipy.integrate.trapz(self(ages) * density, ages)
        density_total = scipy.integrate.trapz(density, ages)
        return rate_total / density_total
