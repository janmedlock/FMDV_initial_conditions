'''Death rate.'''

import numpy
import pandas
import scipy.integrate

from . import _population


class DeathRate:
    '''Death rate.'''

    # Annual survival in age bands (ages in years).
    _annual_survival = {(0, 1): 0.66,
                        (1, 3): 0.79,
                        (3, 12): 0.88,
                        (12, numpy.inf): 0.66}
    _annual_survival = pandas.Series(
        _annual_survival.values(),
        index=pandas.IntervalIndex.from_tuples(_annual_survival.keys(),
                                               closed='left')
    )

    # Death rate with units per year.
    _death_rate = - numpy.log(_annual_survival)

    def __init__(self, parameters):
        # The death rate does not depend on `parameters`.
        pass

    def __call__(self, age):
        '''Death rate.'''
        out = self._death_rate[age]
        try:
            out.set_axis(age, inplace=True)
        except AttributeError:
            pass
        return out

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
