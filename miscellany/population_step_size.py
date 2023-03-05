#!/usr/bin/python3
'''Test the population model.'''

import numpy

from context import models
from models.age_structured import _population


def stable_age_density(parameters, t_step):
    '''Get the stable age density.'''
    model = _population.Model(parameters.birth,
                              parameters.death,
                              t_step=t_step)
    return model.stable_age_density()


def coarsen(a0, n0):
    '''Coarsen the solution `(a0, n0)` onto a grid of every other step
    in `a0`. When `a0` has constant step size, this doubles the step
    size.'''
    assert len(a0) % 2 == 1
    a1 = a0[::2]
    n1 = (numpy.hstack([n0[:-1].reshape((-1, 2)).sum(axis=1),
                        n0[-1]])
          / 2)
    return (a1, n1)

def error(parameters, t_step):
    (a1, n1) = stable_age_density(parameters,
                                  t_step=t_step)
    # Run with time step `t_step / 2` and coarsen to the the same ages
    # as `n1`.
    (a0, n0) = coarsen(*stable_age_density(parameters,
                                           t_step=(t_step / 2)))
    assert numpy.allclose(a0, a1)
    err_rel = (n0 - n1) / n0
    return numpy.linalg.norm(err_rel, ord=numpy.inf)

if __name__ == '__main__':
    parameters = models.parameters.ModelParametersAgeDependent()
    t_step = 1e-2
    err = error(parameters, t_step)
    print(f'{t_step=}: {err=}')
