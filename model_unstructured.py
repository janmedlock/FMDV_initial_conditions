#!/usr/bin/python3
'''Based on our FMDV work, this is an unstructured model with
periodic birth rate.'''

import matplotlib.pyplot
import numpy

import models


if __name__ == '__main__':
    (t_start, t_end) = (0, 10)
    period = model.parameters.period
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_constant = models.unstructured.Model(birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end))
    ax_solution = models.plotting.solution(solution_constant)
    equilibrium = model_constant.find_equilibrium(
        solution_constant.loc[t_end]
    )
    ax_state = models.plotting.state(equilibrium, states=plot_states)
    equilibrium_eigvals = model_constant.get_eigenvalues(equilibrium)
    equilibrium_mults = numpy.exp(equilibrium_eigvals * period)
    ax_mults = models.plotting.multipliers(equilibrium_mults,
                                           label='equilibrium')

    model = models.unstructured.Model()
    solution = model.solve((t_start, t_end))
    models.plotting.solution(solution, ax=ax_solution, legend=False)
    limit_cycle = model.find_limit_cycle(period, t_end % period,
                                         solution.loc[t_end])
    models.plotting.state(limit_cycle, states=plot_states, ax=ax_state)
    limit_cycle_mults = model.get_characteristic_multipliers(limit_cycle)
    models.plotting.multipliers(limit_cycle_mults, label='limit cycle',
                                legend=True, ax=ax_mults)

    matplotlib.pyplot.show()
