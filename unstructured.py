#!/usr/bin/python3
'''Based on our FMDV work, this is an unstructured model with
periodic birth rate.'''

import matplotlib.pyplot

import models


if __name__ == '__main__':
    (t_start, t_end) = (0, 10)
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_constant = models.unstructured.Model(birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end))
    ax_solution = solution_constant.plotting.solution()
    equilibrium = model_constant.find_equilibrium(solution_constant.loc[t_end])
    ax_state = equilibrium.plotting.state(states=plot_states)
    equilibrium_eigvals = model_constant.get_eigenvalues(equilibrium)
    print(equilibrium_eigvals)

    model = models.unstructured.Model()
    solution = model.solve((t_start, t_end))
    solution.plotting.solution(ax=ax_solution, legend=False)
    limit_cycle = model.find_limit_cycle(model.birth.period, t_end,
                                         solution.loc[t_end])
    limit_cycle.plotting.state(states=plot_states, ax=ax_state)
    limit_cycle_eigvals = model.get_characteristic_exponents(limit_cycle)
    print(limit_cycle_eigvals)

    matplotlib.pyplot.show()
