#!/usr/bin/python3

import matplotlib.pyplot
import numpy

from context import models


if __name__ == '__main__':
    params = dict(birth_variation=0)
    (t_start, t_end) = (0, 10)
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_us = models.unstructured.Model(**params)
    solution_us = model_us.solve((t_start, t_end))
    ax_solution = models.plotting.solution(solution_us)
    equilibrium_us = model_us.find_equilibrium(solution_us.loc[t_end])
    ax_state = models.plotting.state(equilibrium_us,
                                     label='unstructured',
                                     states=plot_states)

    model_tses = models.time_since_entry_structured.Model(**params)
    solution_tses = model_tses.solve((t_start, t_end))
    models.plotting.solution(model_tses.integral_over_z(solution_tses),
                             ax=ax_solution, legend=False)
    equilibrium_tses = model_tses.find_equilibrium(solution_tses.loc[t_end],
                                                   options=dict(disp=True))
    models.plotting.state(model_tses.integral_over_z(equilibrium_tses),
                          label='time-since-entry structured',
                          states=plot_states,
                          ax=ax_state)

    model_as = models.age_structured.Model(**params)
    solution_as = model_as.solve((t_start, t_end))
    models.plotting.solution(model_as.integral_over_a(solution_as),
                             ax=ax_solution, legend=False)
    equilibrium_as = model_as.find_equilibrium(solution_as.loc[t_end],
                                               options=dict(disp=True))
    models.plotting.state(model_as.integral_over_a(equilibrium_as),
                          label='age structured',
                          states=plot_states,
                          ax=ax_state)

    ax_state.legend()
    matplotlib.pyplot.show()
