#!/usr/bin/python3

import matplotlib.pyplot
import numpy

from context import models
import models.age_structured
import models.time_since_entry_structured
import models.unstructured


if __name__ == '__main__':
    params = dict(birth_variation=0)
    (t_start, t_end) = (0, 1)
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_us = models.unstructured.Model(**params)
    solution_us = model_us.solve((t_start, t_end))
    ax_solution = solution_us.plotting.solution()
    equilibrium_us = model_us.find_equilibrium(solution_us.loc[t_end])
    ax_state = equilibrium_us.plotting.state(label='unstructured',
                                             states=plot_states)

    model_tses = models.time_since_entry_structured.Model(**params)
    y_start_tses = model_tses.initial_conditions_from_unstructured(
        equilibrium_us)
    solution_tses = model_tses.solve((t_start, t_end),
                                     y_start=y_start_tses)
    solution_tses.time_since_entry.aggregate() \
                 .plotting.solution(ax=ax_solution, legend=False)
    equilibrium_tses = model_tses.find_equilibrium(y_start_tses,
                                                   options=dict(disp=True))
    equilibrium_tses.time_since_entry.aggregate() \
                    .plotting.state(label='time-since-entry structured',
                                    states=plot_states,
                                    ax=ax_state)

    model_as = models.age_structured.Model(**params)
    y_start_as = model_as.initial_conditions_from_unstructured(
        equilibrium_us)
    solution_as = model_as.solve((t_start, t_end),
                                 y_start=y_start_as)
    solution_as.age.aggregate() \
               .plotting.solution(ax=ax_solution, legend=False)
    equilibrium_as = model_as.find_equilibrium(y_start_as,
                                               options=dict(disp=True))
    equilibrium_as.age.aggregate() \
                  .plotting.state(label='age structured',
                                  states=plot_states,
                                  ax=ax_state)

    ax_state.legend()
    matplotlib.pyplot.show()
