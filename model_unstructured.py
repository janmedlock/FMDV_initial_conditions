#!/usr/bin/python3
'''Based on our FMDV work, this is an unstructured model with
periodic birth rate.'''

import collections
import itertools

import matplotlib.pyplot
import matplotlib.rcsetup
import numpy

import models


Model = models.unstructured.Model

PLOT_STATES = ['susceptible', 'infectious', 'recovered']


def _get_period(parameters):
    parameters_birth_nonconstant = parameters.copy()
    try:
        del parameters_birth_nonconstant['birth_variation']
    except KeyError:
        pass
    return models.parameters.Parameters(**parameters_birth_nonconstant) \
                            .birth_period


def run_sat(SAT, birth_constant,
            t_start=0, t_end=20,
            plot_solution=True, plot_solution_kwds={},
            plot_limit_set=True, plot_limit_set_kwds={},
            plot_multipliers=True, plot_multipliers_kwds={}):
    '''Run one SAT.'''
    parameters = {
        'SAT': SAT,
    }
    if birth_constant:
        parameters['birth_variation'] = 0
    model = Model(**parameters)
    period = _get_period(parameters)
    solution = model.solve((t_start, t_end))
    if plot_solution:
        if callable(plot_solution_kwds):
            plot_solution_kwds = plot_solution_kwds(SAT, birth_constant)
        models.plotting.solution(solution,
                                 **plot_solution_kwds)
    if birth_constant:
        limit_set = model.find_equilibrium(solution.loc[t_end])
    else:
        limit_set = model.find_limit_cycle(period,
                                           t_end % period,
                                           solution.loc[t_end])
    if plot_limit_set:
        if callable(plot_limit_set_kwds):
            plot_limit_set_kwds = plot_limit_set_kwds(SAT, birth_constant)
        models.plotting.state(limit_set,
                              **plot_limit_set_kwds)
    if birth_constant:
        exponents = model.get_eigenvalues(limit_set)
        multipliers = numpy.exp(exponents * period)
    else:
        multipliers = model.get_characteristic_multipliers(limit_set)
    if plot_multipliers:
        if callable(plot_multipliers_kwds):
            plot_multipliers_kwds = plot_multipliers_kwds(SAT, birth_constant)
        models.plotting.multipliers(multipliers,
                                    **plot_multipliers_kwds)


def plot_solution_kwds_build(SATs, birth_constants):
    prop_cycle = models.plotting.solution_prop_cycle(Model.states)
    subplot_kw = {
        'prop_cycle': prop_cycle,
    }
    (fig, axes) = matplotlib.pyplot.subplots(
        nrows=len(SATs),
        squeeze=False,
        sharex='col',
        subplot_kw=subplot_kw,
        layout='constrained',
    )
    axes = numpy.squeeze(axes, axis=1)
    axes = dict(zip(SATs, axes))
    for (SAT, ax) in axes.items():
        ax.set_ylabel(f'SAT{SAT}')

    def plot_solution_kwds(SAT, birth_constant):
        ax = axes[SAT]
        legend = (ax.get_subplotspec().is_first_row()
                  & birth_constant)
        return {
            'ax': ax,
            'legend': legend,
        }

    return plot_solution_kwds


def plot_limit_set_kwds_build(SATs, birth_constants):
    fig = matplotlib.pyplot.figure(
        layout='constrained',
    )
    state_prop_cycle = models.plotting.state_prop_cycle()
    birth_style = {
        True: {
            'linestyle': 'none',
        },
        False: {
            'marker': 'none',
        },
    }
    prop_cycle = collections.defaultdict(list)
    for state_prop in state_prop_cycle:
        for birth_constant in birth_constants:
            style = state_prop | birth_style[birth_constant]
            for (k, v) in style.items():
                prop_cycle[k].append(v)
    prop_cycle = matplotlib.rcsetup.cycler(**prop_cycle)
    ax = models.plotting.state_make_axes(
        PLOT_STATES,
        fig=fig,
        prop_cycle=prop_cycle,
    )

    def plot_limit_set_kwds(SAT, birth_constant):
        label = f'SAT{SAT}' if birth_constant else ''
        return {
            'states': PLOT_STATES,
            'ax': ax,
            'label': label,
        }

    return plot_limit_set_kwds


def plot_multipliers_kwds_build(SATs, birth_constants):
    fig = matplotlib.pyplot.figure(
        layout='constrained',
    )
    colors = models.plotting.colors[:len(SATs)]
    markers = models.plotting.markers[:len(birth_constants)]
    prop_cycle = colors * markers
    ax = models.plotting.multipliers_make_axes(
        fig=fig,
        prop_cycle=prop_cycle,
    )

    def plot_multipliers_kwds(SAT, birth_constant):
        label = (
            f'SAT{SAT} '
            + ('equilibrium' if birth_constant else 'limit cycle')
        )
        return {
            'ax': ax,
            'label': label,
        }

    return plot_multipliers_kwds


if __name__ == '__main__':
    SATs = (1, 2, 3)
    birth_constants = (True, False)

    plot_solution_kwds = plot_solution_kwds_build(SATs, birth_constants)
    plot_limit_set_kwds = plot_limit_set_kwds_build(SATs, birth_constants)
    plot_multipliers_kwds = plot_multipliers_kwds_build(SATs, birth_constants)

    for SAT in SATs:
        for birth_constant in birth_constants:
            run_sat(SAT, birth_constant,
                    plot_solution_kwds=plot_solution_kwds,
                    plot_limit_set_kwds=plot_limit_set_kwds,
                    plot_multipliers_kwds=plot_multipliers_kwds)

    matplotlib.pyplot.show()
