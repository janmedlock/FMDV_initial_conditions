#!/usr/bin/python3
#
#                                                CSR       CSC
# Constant birth
#   population_growth_rate()                     0.2 sec   0.2 sec
#   birth_scaling_for_zero_population_growth()   1.7 sec   1.8 sec
#   stable_age_density()                         0.2 sec   0.2 sec
# Periodic birth
#   population_growth_rate()                     4.0 sec   8.3 sec
#   birth_scaling_for_zero_population_growth()  35.4 sec  89.6 sec
#   stable_age_density()                         4.1 sec   9.1 sec
#

import scipy.sparse

from context import models
import timer


# Monkeypatch to set the array type.
models._utility.sparse.array = scipy.sparse.csr_array

Solver = models.age_structured._population._solver.Solver

# Monkeypatch to remove caching.
Solver._init_cached = lambda self: None

methods = ('population_growth_rate',
           'birth_scaling_for_zero_population_growth',
           'stable_age_density')
for name in methods:
    method = getattr(Solver, name)
    setattr(Solver, name + '_t', timer.timer(method))


def get_solver(**kwds):
    parameters = models.parameters.Parameters(**kwds)
    birth = models.birth.Birth(parameters)
    death = models.death.Death(parameters)
    t_step = models.age_structured._population.Model._t_step_default
    a_max = models.age_structured._age.max_default
    return Solver(birth, death, t_step, a_max)


def test_solver(solver):
    solver._monodromy_init()
    solver.population_growth_rate_t(1, _guess=0)
    solver.birth.mean *= solver.birth_scaling_for_zero_population_growth_t()
    solver.stable_age_density_t()


if __name__ == '__main__':
    solver_constant = get_solver(birth_variation=0)
    test_solver(solver_constant)

    solver = get_solver()
    test_solver(solver)
