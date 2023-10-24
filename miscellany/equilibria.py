#!/usr/bin/python3
'''Find equilibria for complex models by using the equilibrium from
simpler models.'''

import matplotlib.pyplot
import numpy

from context import models


if __name__ == '__main__':
    params = {'birth_variation': 0}
    T_SOLVE = 20
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_us = models.unstructured.Model(**params)
    ic_us = model_us.build_initial_conditions()
    eql_us = model_us.find_equilibrium(ic_us,
                                       t_solve=T_SOLVE)
    ax_state = models.plotting.state(eql_us,
                                     label='unstructured',
                                     states=plot_states)

    model_tses = models.time_since_entry_structured.Model(**params)
    n_z = model_tses._survivals_scaled()
    ic_tses = eql_us * n_z
    eql_tses = model_tses.find_equilibrium(ic_tses,
                                           display=True)
    models.plotting.state(model_tses.integral_over_z(eql_tses),
                          label='time-since-entry structured',
                          states=plot_states,
                          ax=ax_state)

    model_as = models.age_structured.Model(**params)
    n_a = model_as.stable_age_density()
    ic_as = numpy.outer(eql_us, n_a).ravel()
    # The equilbrium found has a negative component.
    eql_as = model_as.find_equilibrium(ic_as,
                                       display=True)
    models.plotting.state(model_as.integral_over_a(eql_as),
                          label='age structured',
                          states=plot_states,
                          ax=ax_state)

    ax_state.legend()
    matplotlib.pyplot.show()
