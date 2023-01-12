#!/usr/bin/python3
'''Based on our FMDV work, this is an unstructured model with
periodic birth rate.'''

import matplotlib.pyplot

import models.unstructured


if __name__ == '__main__':
    (t_start, t_end, t_step) = (0, 10, 0.001)
    state_states = ['susceptible', 'infectious', 'recovered']

    model_constant = models.unstructured.Model(birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end), t_step)
    ax_solution = solution_constant.solution.plot_solution()
    equilibrium = model_constant.find_equilibrium(solution_constant.loc[t_end])
    ax_state = equilibrium.solution.plot_state(states=state_states)
    equilibrium_eigvals = model_constant.get_eigenvalues(equilibrium)
    print(equilibrium_eigvals)

    model = models.unstructured.Model()
    solution = model.solve((t_start, t_end), t_step)
    solution.solution.plot_solution(ax=ax_solution, legend=False)
    limit_cycle = model.find_limit_cycle(model.parameters.birth_period,
                                         t_end, t_step,
                                         solution.loc[t_end])
    limit_cycle.solution.plot_state(states=state_states,
                                    ax=ax_state)
    limit_cycle_eigvals = model.get_characteristic_exponents(limit_cycle)
    print(limit_cycle_eigvals)

    matplotlib.pyplot.show()
