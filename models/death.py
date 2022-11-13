'''Death.'''

import numpy
import scipy.integrate

from . import _population


class Death:
    '''Death.'''

    # Annual survival in age bands (ages in years).
    # The bands are closed on the left, open on the right.
    _annual_survival = {(0, 1): 0.66,
                        (1, 3): 0.79,
                        (3, 12): 0.88,
                        (12, numpy.PINF): 0.66}

    # Death rate values with units per year.
    _rate = {key: -numpy.log(val)
             for (key, val)
             in _annual_survival.items()}

    def __init__(self, parameters):
        # Death does not depend on `parameters`.
        pass

    def rate(self, age):
        '''Death rate.'''
        # `condlist` is a list of arrays of whether `age` is in each
        # interval in `self._death_rate`.
        condlist = [(left <= age) & (age < right)
                    for (left, right) in self._rate.keys()]
        return numpy.select(condlist, self._rate.values(), numpy.nan)

    def rate_population_mean(self, birth, *args, **kwds):
        '''Get the mean death rate when the population is at the stable
        age density.'''
        (ages, density) = _population.stable_age_density(birth, self,
                                                         *args, **kwds)
        rate_total = scipy.integrate.trapz(self.rate(ages) * density, ages)
        density_total = scipy.integrate.trapz(density, ages)
        return rate_total / density_total
