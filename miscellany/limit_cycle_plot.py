#!/usr/bin/python3
'''Based on our FMDV work, ...'''

import matplotlib.pyplot
import pandas

import limit_cycle


def _integrate(model, lcy):
    integrals = ('integral_over_a', 'integral_over_z')
    for attr in integrals:
        try:
            integral = getattr(model, attr)
        except AttributeError:
            pass
        else:
            lcy = integral(lcy)
    return lcy


def _plot(ax, Model, SAT, color, linestyle):
    model = Model(SAT=SAT)
    lcy = pandas.read_pickle(limit_cycle.filename(model, SAT))
    lcy_integral = _integrate(model, lcy)
    label = f'SAT{SAT}, {model.variables} structured'
    lcy_integral.loc[:, 'infectious'] \
                .plot(ax=ax, color=color, linestyle=linestyle, label=label)


def plot_age(ax, SAT, color):
    return _plot(ax, limit_cycle.ModelAge, SAT, color, 'solid')


def plot_tse(ax, SAT, color):
    return _plot(ax, limit_cycle.ModelTSE, SAT, color, 'dashed')


def plot():
    (_, ax) = matplotlib.pyplot.subplots(layout='constrained')
    for SAT in (1, 2, 3):
        color = f'C{SAT-1}'
        plot_age(ax, SAT, color)
        try:
            plot_tse(ax, SAT, color)
        except FileNotFoundError:
            pass
    ax.legend()


if __name__ == '__main__':
    plot()
    matplotlib.pyplot.show()
