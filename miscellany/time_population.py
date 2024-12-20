#!/usr/bin/python3
'''Time the population solver.

+----------------------------------------------+----------+----------+
|                                              |   jacobian_method   |
| birth rate & method                          +----------+----------+
|                                              |   CSR    |    CSC   |
+----------------------------------------------+----------+----------+
| Constant birth                               |          |          |
|   population_growth_rate()                   |  0.2 sec |  0.2 sec |
|   birth_scaling_for_zero_population_growth() |  1.7 sec |  1.8 sec |
|   stable_age_density()                       |  0.2 sec |  0.2 sec |
+----------------------------------------------+----------+----------+
| Periodic birth                               |          |          |
|   population_growth_rate()                   |  4.0 sec |  8.3 sec |
|   birth_scaling_for_zero_population_growth() | 35.4 sec | 89.6 sec |
|   stable_age_density()                       |  4.1 sec |  9.1 sec |
+----------------------------------------------+----------+----------+
'''


import scipy.sparse

from context import models
from models import _utility
from models.age_structured import _population
import timer


# Monkeypatch to set the array type.
_utility.sparse.Array = scipy.sparse.csr_array


# pylint: disable-next=too-few-public-methods
class TestSolver(_population.solver.Solver):
    '''Test the solver.'''

    _methods_timed = ('population_growth_rate',
                      'birth_scaling_for_zero_population_growth',
                      'stable_age_density')

    # pylint: disable-next=too-few-public-methods
    class Parameters(models.parameters.PopulationParameters):
        '''Simplified model parameters.'''

        def __init__(self, **kwds):
            parameters = models.parameters.Parameters(**kwds)
            # Set birth mean without finding the value that gives
            # zero population growth rate.
            parameters.birth.mean = 1.
            super().__init__(parameters)

    def _cache_methods(self):
        '''Monkeypatch to disable caching.'''

    def _time_methods(self):
        '''Time the methods in `_methods_timed`.'''
        for name in self._methods_timed:
            method = getattr(self, name)
            timed = timer.timer(method)
            setattr(self, name, timed)

    def __init__(self, **kwds):
        parameters = self.Parameters(**kwds)
        model = _population.model.Model(parameters=parameters)
        super().__init__(model)
        self._check_matrices()
        self._time_methods()

    def test(self):
        '''Test the simplified model.'''
        self.population_growth_rate(1., _guess=0)
        bscl = self.birth_scaling_for_zero_population_growth()
        self.parameters.birth.mean *= bscl
        self.stable_age_density()


if __name__ == '__main__':
    solver_constant = TestSolver(birth_variation=0)
    solver_constant.test()

    solver = TestSolver()
    solver.test()
