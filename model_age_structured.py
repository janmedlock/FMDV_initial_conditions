#!/usr/bin/python3
'''Based on our FMDV work, this is an age-structured model with
periodic birth rate.'''

import matplotlib.pyplot
import numpy

import models


if __name__ == '__main__':
    Model = models.age_structured.Model
    (t_start, t_end) = (0, 10)
    plot_states = ['susceptible', 'infectious', 'recovered']
    parameters = {'SAT': 1}

    model_const = Model(**parameters, birth_variation=0)
    model = Model(**parameters)
    period = model.parameters.period

    soln_const = model_const.solve((t_start, t_end))
    ax_soln = models.plotting.solution(model_const.integral_over_a(soln_const))
    soln = model.solve((t_start, t_end))
    models.plotting.solution(model.integral_over_a(soln),
                             ax=ax_soln, legend=False)

    eql = model_const.find_equilibrium(soln_const.loc[t_end])
    ax_state = models.plotting.state(model_const.integral_over_a(eql),
                                     states=plot_states)
    lcy = model.find_limit_cycle(period, t_end % period, soln.loc[t_end])
    models.plotting.state(model.integral_over_a(lcy), ax=ax_state)

    exps_eql = model_const.get_eigenvalues(eql)
    mults_eql = numpy.exp(exps_eql * period)
    ax_mults = models.plotting.multipliers(mults_eql, label='equilibrium')
    mults_lcy = model.get_characteristic_multipliers(lcy, display=True)
    models.plotting.multiprocessing(mults_lcy, label='limit cycle',
                                    legend=True, ax=ax_mults)

    matplotlib.pyplot.show()
