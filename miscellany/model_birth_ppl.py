#!/usr/bin/python3
'''Test the periodic piecewise-linear birth rate.'''

import matplotlib.pyplot
import numpy

from context import models


models.birth.BirthPeriodic = models.birth.BirthPeriodicPiecewiseLinear

Model = models.unstructured.Model


if __name__ == '__main__':
    (t_start, t_end) = (0, 20)
    plot_states = ['susceptible', 'infectious', 'recovered']
    parameters = {'SAT': 3}

    model_const = Model(**parameters, birth_variation=0)
    model = Model(**parameters)
    period = model.parameters.period

    soln_const = model_const.solve((t_start, t_end))
    ax_soln = models.plotting.solution(soln_const)
    soln = model.solve((t_start, t_end))
    models.plotting.solution(soln, ax=ax_soln, legend=False)

    eql = model_const.find_equilibrium(soln_const.loc[t_end])
    ax_state = models.plotting.state(eql, states=plot_states)
    lcy = model.find_limit_cycle(period, t_end % period, soln.loc[t_end],
                                 method='broyden1',
                                 options={'maxiter': 1200})
    models.plotting.state(lcy, ax=ax_state)

    exps_eql = model_const.get_eigenvalues(eql)
    mults_eql = numpy.exp(exps_eql * period)
    ax_mults = models.plotting.multipliers(mults_eql, label='equilibrium')
    mults_lcy = model.get_characteristic_multipliers(lcy)
    models.plotting.multipliers(mults_lcy, label='limit cycle',
                                legend=True, ax=ax_mults)

    matplotlib.pyplot.show()
