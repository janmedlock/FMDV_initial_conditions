#!/usr/bin/python3
'''Based on our FMDV work, this is a time-since-entry-structured model
with periodic birth rate.'''

import matplotlib.pyplot

import models


if __name__ == '__main__':
    (t_start, t_end) = (0, 10)
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_constant = models.time_since_entry_structured.Model(
        birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end))
    ax_solution = models.plotting.solution(
        model_constant.integral_over_z(solution_constant)
    )
    equilibrium = model_constant.find_equilibrium(solution_constant.loc[t_end])
    ax_state = models.plotting.state(
        model_constant.integral_over_z(equilibrium),
        states=plot_states
    )
    equilibrium_eigvals = model_constant.get_eigenvalues(equilibrium)
    print(equilibrium_eigvals)

    model = models.time_since_entry_structured.Model()
    solution = model.solve((t_start, t_end))
    models.plotting.solution(model.integral_over_z(solution),
                             ax=ax_solution, legend=False)

    matplotlib.pyplot.show()
