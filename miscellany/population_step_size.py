#!/usr/bin/python3
'''Test the population model.'''

import numpy

from context import models
import models.age_structured._population


def stable_age_density(parameters, t_step):
    '''Get the stable age density.'''
    model = models.age_structured._population.Model(parameters.birth,
                                                    parameters.death,
                                                    t_step=t_step)
    return model.stable_age_density()


def coarsen(ages, density):
    '''Coarsen the solution `(ages, density)` onto a grid with double
    the step size of `ages`.'''
    assert len(ages) % 2 == 1
    ages_coarse = ages[::2]
    density_coarse = 0.5 * numpy.hstack([
        density[:-1].reshape((-1, 2)).sum(axis=1),
        density[-1]
    ])
    return (ages_coarse, density_coarse)


def get_error(t_step):
    '''Get the maximum absolute relative error in the stable age
    distribution with time step `t_step` from the stable age
    distribution with time step `t_step` / 2.'''
    parameters = models.parameters.ModelParametersAgeDependent()
    # Run with time step `t_step`.
    (ages, density) = stable_age_density(parameters, t_step=t_step)
    # Run with time step `t_step / 2` and coarsen to `ages`.
    t_step_fine = t_step / 2
    (ages_fine, density_fine) = stable_age_density(parameters,
                                                   t_step=t_step_fine)
    (ages_fine_coarse, density_fine_coarse) = coarsen(ages_fine, density_fine)
    assert numpy.allclose(ages_fine_coarse, ages)
    error_relative = (density - density_fine_coarse) / density_fine_coarse
    return numpy.linalg.norm(error_relative, ord=numpy.inf)


if __name__ == '__main__':
    t_step = 1e-2
    error = get_error(t_step)
    print(f'{t_step=}: {error=}')
