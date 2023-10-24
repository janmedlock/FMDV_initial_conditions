#!/usr/bin/python3

import time

import matplotlib.pyplot
import numpy
import pandas
import scipy.stats

from context import models


def time_solve(Model, t_step, t_span,
               display=True, check_nonnegative=False,
               **kwds):
    model = Model(t_step=t_step, **kwds)
    y_start = model.build_initial_conditions()
    start = time.perf_counter()
    y_end = model.solution_at_t_end(t_span,
                                    y_start=y_start,
                                    display=display,
                                    check_nonnegative=check_nonnegative)
    duration = time.perf_counter() - start
    return duration


def time_solves(Model, t_steps, t_span, **kwds):
    times = [
        time_solve(Model, t_step, t_span, **kwds)
        for t_step in t_steps
    ]
    index = pandas.Index(t_steps, name='t_step')
    name = 'run time (sec)'
    return pandas.Series(times, index=index, name=name)


def plot_times(times, t_step_default):
    axes = times.plot(ylabel=times.name, marker='o', logx=True, logy=True)
    axes.invert_xaxis()
    x = numpy.log(times.index)
    y = numpy.log(times)
    model = scipy.stats.linregress(x, y)
    x_ = numpy.linspace(times.index.max(), t_step_default, 101)
    y_ = numpy.exp(model.slope * numpy.log(x_) + model.intercept)
    axes.plot(x_, y_, color='black', linestyle='dashed', zorder=1)
    return axes


if __name__ == '__main__':
    Model = models.combination.Model
    t_steps = [1e-1, 5e-2, 2e-2, 1e-2]
    t_span = (0, 1)
    kwds = {'transmission_rate': 10,
            '_solver_options': {'_check_matrices': False}}
    times = time_solves(Model, t_steps, t_span, **kwds)
    axes = plot_times(times, Model._t_step_default)
    matplotlib.pyplot.show()
