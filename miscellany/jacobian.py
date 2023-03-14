#!/usr/bin/python3

from context import models


if __name__ == '__main__':
    Model = models.age_structured.Model

    (t_start, t_end) = (0, 10)

    model_constant = Model(birth_variation=0)
    equilibrium = model_constant.find_equilibrium(
        model_constant.build_initial_conditions(), t_solve=t_end
    )
    print('Starting `jacobian()`.')
    J = model_constant._solver.jacobian(t_start, equilibrium, equilibrium)
    print('Finished `jacobian()`.')
    # equilibrium_eigvals = model_constant.get_eigenvalues(equilibrium)

    # model = Model()
    # solution = model.solve((t_start, t_end))
    # limit_cycle = model.find_limit_cycle(model.parameters.period,
    #                                      t_end % model.parameters.period,
    #                                      solution.loc[t_end])
    # limit_cycle_eigvals = model.get_characteristic_exponents(limit_cycle,
    #                                                          display=True)
