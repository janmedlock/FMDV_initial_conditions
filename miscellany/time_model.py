#!/usr/bin/python3
'''Time model methods vs. step size.

+-----------------------------------+------------+------------+
|                                   |     jacobian_method     |
| model & method                    +------------+------------+
|                                   |    CSR     |    CSC     |
+-----------------------------------+------------+------------+
| time-since-entry-structured model |            |            |
|   get_characteristic_exponents()  | 8006.6 sec | 7800.1 sec |
+-----------------------------------+------------+------------+
| age-structured model              |            |            |
|   solve() unstructured            |  320.0 sec |            |
|   find_equilibrium()              |    1.2 sec |            |
|   get_eigenvalues()               |            |            |
|   find_limit_cycle()              |            |            |
|   get_characteristic_exponents()  |            |            |
+-----------------------------------+------------+------------+
'''

from context import models
import timer


_methods_to_time = ('solve', 'find_equilibrium', 'get_eigenvalues',
                    'find_limit_cycle', 'get_characteristic_exponents')


def time_model(Model):
    '''Add timers to some methods of `Model`.'''
    # Only add timers once.
    if not hasattr(Model, '_timed'):
        for name in _methods_to_time:
            setattr(Model, name, timer.timer(getattr(Model, name)))
        Model._timed = True


if __name__ == '__main__':
    Model = models.age_structured.Model
    time_model(Model)

    (t_start, t_end) = (0, 10)

    model_constant = Model(birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end))
    equilibrium = model_constant.find_equilibrium(
        solution_constant.loc[t_end]
    )
    equilibrium_eigvals = model_constant.get_eigenvalues(equilibrium)

    model = Model()
    solution = model.solve((t_start, t_end))
    limit_cycle = model.find_limit_cycle(model.parameters.period,
                                         t_end % model.parameters.period,
                                         solution.loc[t_end])
    limit_cycle_eigvals = model.get_characteristic_exponents(limit_cycle,
                                                             display=True)
