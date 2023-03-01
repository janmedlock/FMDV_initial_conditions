#!/usr/bin/python3
'''Based on our FMDV work, this is an age-structured model with
periodic birth rate.'''

import matplotlib.pyplot

import models


if __name__ == '__main__':
    (t_start, t_end) = (0, 1)
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_constant = models.age_structured.Model(birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end))
    ax_solution = model_constant.integral_over_a(solution_constant) \
                                .plotting.solution()
    equilibrium = model_constant.find_equilibrium(solution_constant.loc[t_end])
    ax_state = model_constant.integral_over_a(equilibrium) \
                             .plotting.state(states=plot_states)

    model = models.age_structured.Model()
    solution = model.solve((t_start, t_end))
    model.integral_over_a(solution) \
         .plotting.solution(ax=ax_solution, legend=False)

    matplotlib.pyplot.show()
