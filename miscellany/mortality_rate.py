#!/usr/bin/python3

import numpy
import scipy.integrate

from context import models


def get_death_rate_mean():
    '''Get the mean death rate weighted by the stable age
    distbribution.'''
    death = models.death.Death(parameters={})

    def numerator_integrand(age):
        return death.rate(age) * death.survival(age)

    (numerator, _) = scipy.integrate.quad(numerator_integrand,
                                          0, numpy.inf)
    (denominator, _) = scipy.integrate.quad(death.survival,
                                            0, numpy.inf)
    return numerator / denominator


if __name__ == '__main__':
    death_rate_mean = get_death_rate_mean()
    print(death_rate_mean)

    model = models.unstructured.Model(birth_variation=0)
    print(model.parameters.death_rate_mean)
