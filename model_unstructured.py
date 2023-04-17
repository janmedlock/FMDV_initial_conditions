#!/usr/bin/python3
'''Based on our FMDV work, this is an unstructured model with
periodic birth rate.'''

import matplotlib.pyplot
import numpy

import models


if __name__ == '__main__':
    Model = models.unstructured.Model
    (t_start, t_end) = (0, 10)
    plot_states = ['susceptible', 'infectious', 'recovered']

    model_const = Model(birth_variation=0)
    model = Model()
    period = model.parameters.period

    soln_const = model_const.solve((t_start, t_end))
    ax_soln = models.plotting.solution(soln_const)
    soln = model.solve((t_start, t_end))
    models.plotting.solution(soln, ax=ax_soln, legend=False)

    eql = model_const.find_equilibrium(soln_const.loc[t_end])
    ax_state = models.plotting.state(eql, states=plot_states)
    lcy = model.find_limit_cycle(period, t_end % period, soln.loc[t_end])
    models.plotting.state(lcy, ax=ax_state)

    exps_eql = model_const.get_eigenvalues(eql)
    mults_eql = numpy.exp(exps_eql * period)
    ax_mults = models.plotting.multipliers(mults_eql, label='equilibrium')
    mults_lcy = model.get_characteristic_multipliers(lcy)
    models.plotting.multipliers(mults_lcy, label='limit cycle',
                                legend=True, ax=ax_mults)

    matplotlib.pyplot.show()
