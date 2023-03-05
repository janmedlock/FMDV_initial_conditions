'''Death.'''

import numpy

from . import _utility


class Death:
    '''Death.'''

    # Annual survival in age bands (ages in years).
    # The bands are closed on the left, open on the right.
    _annual_survival = {(0, 1): 0.66,
                        (1, 3): 0.79,
                        (3, 12): 0.88,
                        (12, numpy.PINF): 0.66}

    assert numpy.all(0 <= val <= 1
                     for val in _annual_survival.values())

    (_left, _right) = numpy.array(list(zip(*_annual_survival.keys())))

    assert _utility.numerical.is_increasing(_left)
    assert _utility.numerical.is_increasing(_right)
    assert _left[0] == 0
    assert numpy.all(_left[1:] == _right[:-1])
    assert numpy.isposinf(_right[-1])

    # Death rate values with units per year.
    _rate = - numpy.log(list(_annual_survival.values()))

    def __init__(self, parameters):
        # This class does not depend on `parameters`.
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
        '''Log survival.'''
        age = numpy.asarray(age)
        exposure = (numpy.clip(age[..., None], self._left, self._right)
                    - self._left)
        return numpy.sum(- self._rate * exposure, axis=-1)

    def survival(self, age):
        '''Survival.'''
        return numpy.exp(self.logsurvival(age))

    def rate_population_mean(self, population, **kwds):
        '''Get the mean death rate when the population is at the stable
        age density.'''
        (ages, density) = population.stable_age_density(**kwds)
        rate_total = population.integral_over_a(self.rate(ages) * density)
        density_total = population.integral_over_a(density)
        return rate_total / density_total

    def _age_max(self):
        '''Get the last age where `._annual_survival` changes.'''
        age_max = self._left.max()
        return age_max
