'''Death.'''

import numpy

from ._model import population


class Death:
    '''Death.'''

    # Annual survival in age bands (ages in years).
    # The bands are closed on the left, open on the right.
    _annual_survival = {(0, 1): 0.66,
                        (1, 3): 0.79,
                        (3, 12): 0.88,
                        (12, numpy.PINF): 0.66}

    (_left, _right) = numpy.array(list(zip(*_annual_survival.keys())))

    # Death rate values with units per year.
    _rate = - numpy.log(list(_annual_survival.values()))

    def __init__(self, parameters):
        # Death does not depend on `parameters`.
        pass

    def rate(self, age):
        '''Death rate.'''
        age = numpy.asarray(age)
        isin = ((self._left <= age[..., None])
                & (age[..., None] < self._right))
        return numpy.select(isin.swapaxes(-1, 0),
                            self._rate,
                            numpy.nan)

    def logsurvival(self, age):
        age = numpy.asarray(age)
        exposure = (numpy.clip(age[..., None], self._left, self._right)
                    - self._left)
        return numpy.sum(- self._rate * exposure, axis=-1)

    def survival(self, age):
        return numpy.exp(self.logsurvival(age))

    def rate_population_mean(self, birth, *args, **kwds):
        '''Get the mean death rate when the population is at the stable
        age density.'''
        (ages, density) = population.stable_age_density(birth, self,
                                                        *args, **kwds)
        age_step = numpy.mean(numpy.diff(ages))
        rate_total = (self.rate(ages) * density).sum() * age_step
        density_total = density.sum() * age_step
        return rate_total / density_total
