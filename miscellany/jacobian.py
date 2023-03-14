#!/usr/bin/python3

# models.unstructured.Model()
#       base:  0.0 sec
# sparse_csr:  0.0 sec
# sparse_csc:  0.0 sec
#      dense:  0.0 sec
#
# models.time_since_entry_structured.Model()
#       base:  3.9 sec
# sparse_csc:  3.8 sec
# sparse_csr:  3.6 sec
#      dense: 29.2 sec
#
# models.age_structured.Model(t_step=0.01, transmission_rate=10)
# sparse_csc: 42.8 sec
# sparse_csr: 44.5 sec
#      dense:  9.2 sec
#

from context import models

import timer


if __name__ == '__main__':
    # Model = models.unstructured.Model
    # Model = models.time_since_entry_structured.Model
    Model = models.age_structured.Model
    kwds = dict(
        t_step=0.01,
        transmission_rate=10,
        _solver_options=dict(_check_matrices=False,
                             _jacobian_method='base'),
    )

    (t_start, t_end) = (0, 10)

    model_constant = Model(birth_variation=0, **kwds)
    equilibrium = model_constant.find_equilibrium(
        model_constant.build_initial_conditions(), t_solve=t_end
    )
    J = timer.timer(model_constant._solver.jacobian)(t_start,
                                                     equilibrium,
                                                     equilibrium)
    # equilibrium_eigvals = model_constant.get_eigenvalues(equilibrium)
    # print(equilibrium_eigvals)

    # model = Model()
    # solution = model.solve((t_start, t_end))
    # limit_cycle = model.find_limit_cycle(model.parameters.period,
    #                                      t_end % model.parameters.period,
    #                                      solution.loc[t_end])
    # limit_cycle_eigvals = model.get_characteristic_exponents(limit_cycle,
    #                                                          display=True)
