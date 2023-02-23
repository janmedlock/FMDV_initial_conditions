#!/usr/bin/python3
'''Based on our FMDV work, this is a time-since-entry-structured model
with periodic birth rate.'''

import matplotlib.pyplot

import models
import models.time_since_entry_structured


if __name__ == '__main__':
    (t_start, t_end) = (0, 10)
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_constant = models.time_since_entry_structured.Model(
        birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end))
    ax_solution = solution_constant.time_since_entry.aggregate() \
                                   .plotting.solution()
    equilibrium = model_constant.find_equilibrium(solution_constant.loc[t_end])
    ax_state = equilibrium.time_since_entry.aggregate() \
                          .plotting.state(states=plot_states)

    model = models.time_since_entry_structured.Model()
    solution = model.solve((t_start, t_end))
    solution.time_since_entry.aggregate() \
            .plotting.solution(ax=ax_solution, legend=False)

    matplotlib.pyplot.show()
