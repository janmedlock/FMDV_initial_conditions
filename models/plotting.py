'''Plot solutions and points in state space.'''

import functools

import matplotlib.pyplot
import matplotlib.rcsetup


_cmap = matplotlib.pyplot.get_cmap('tab10')

colors = matplotlib.rcsetup.cycler(
    color=_cmap.colors
)

linestyles = matplotlib.rcsetup.cycler(
    linestyle=['solid', 'dotted', 'dashed']
)


markers = matplotlib.rcsetup.cycler(
    marker=['o', '^', 's', '*', '+', 'x']
)


def cycler_inner_product(*cyclers):
    '''Get the inner product of the cyclers, reducing their length to
    that of the smallest one.'''
    # Make them all the same length.
    stop = min(map(len, cyclers))
    cyclers_ = map(lambda x: x[:stop], cyclers)
    # Sum them.
    return functools.reduce(lambda x, y: x + y, cyclers_)


def population_size(obj):
    '''Population size.'''
    # Sum over the last axis.
    axis = obj.ndim - 1
    return obj.sum(axis=axis)


def _get_states(states, obj):
    '''Get the states to plot.'''
    if states is None:
        return obj.axes[-1]
    return states


def state_prop_cycle():
    '''Build a prop_cycle for the state plot.'''
    prop_cycle = cycler_inner_product(colors, markers, linestyles)
    return prop_cycle


def state_make_axes(states, fig=None, prop_cycle=None):
    '''Make state axes.'''
    if len(states) == 2:
        projection = 'rectilinear'
    elif len(states) == 3:
        projection = '3d'
    else:
        raise ValueError(f'{len(states)=}')
    if fig is None:
        fig = matplotlib.pyplot.figure()
    if prop_cycle is None:
        prop_cycle = state_prop_cycle()
    axis_labels = dict(zip(('xlabel', 'ylabel', 'zlabel'), states))
    return fig.add_subplot(projection=projection,
                           prop_cycle=prop_cycle,
                           **axis_labels)


def _get_states_from_axes(ax):
    '''Get the phase states from the axes `ax`.'''
    states = [ax.get_xlabel(),
              ax.get_ylabel()]
    try:
        states.append(ax.get_zlabel())
    except AttributeError:
        pass
    return states


def state(obj, states=None, ax=None, fig=None, legend=True, **kwds):
    '''Make a phase plot.'''
    if ax is None:
        ax = state_make_axes(_get_states(states, obj), fig=fig)
    if states is None:
        states = _get_states_from_axes(ax)
    if obj.ndim == 1:
        ax.plot(*obj[states], **kwds)
    elif obj.ndim == 2:
        # Not `col = obj[states]` because the result must iterate over
        # columns, not rows.
        cols = (obj[col] for col in states)
        ax.plot(*cols, marker='none', **kwds)
    else:
        raise ValueError(f'{obj.ndim=}')
    if legend:
        ax.legend()
    return ax


def solution_prop_cycle(states):
    '''Build a prop_cycle for the solution plot.'''
    inner = colors[:len(states)]
    outer = linestyles
    prop_cycle = outer * inner
    return prop_cycle


def solution_make_axes(states, ax=None, fig=None):
    '''Make solution axes.'''
    if ax is None:
        if fig is None:
            fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot()
    ax.set_prop_cycle(solution_prop_cycle(states))
    return ax


def solution(obj, states=None, ax=None, fig=None, **kwds):
    '''Plot the solution vs time.'''
    states = _get_states(states, obj)
    if ax is None:
        ax = solution_make_axes(states, fig=fig)
    return obj[states].plot(ax=ax, **kwds)


def _complex_prop_cycle():
    '''Build a prop_cycle for the eigenvalue plot.'''
    linestyle = matplotlib.rcsetup.cycler(linestyle=['none'])
    prop_cycle = cycler_inner_product(colors, markers) * linestyle
    return prop_cycle


def _complex_make_axes(title, fig=None, prop_cycle=None):
    if fig is None:
        fig = matplotlib.pyplot.figure()
    if prop_cycle is None:
        prop_cycle = _complex_prop_cycle()
    return fig.add_subplot(
        title=title,
        xlabel='$\\Re$',
        ylabel='$\\Im$',
        prop_cycle=prop_cycle,
    )


def _complex(vals, ax, legend, **kwds):
    ax.plot(vals.real, vals.imag, linestyle='', **kwds)
    if legend:
        ax.legend()
    return ax


def multipliers_make_axes(fig=None, prop_cycle=None):
    title = 'Floquet multipliers'
    return _complex_make_axes(title, fig=fig, prop_cycle=prop_cycle)


def multipliers(mults, ax=None, fig=None, legend=True, **kwds):
    if ax is None:
        ax = multipliers_make_axes(fig=fig)
    return _complex(mults, ax, legend, **kwds)


def exponents_make_axes(fig=None, prop_cycle=None):
    title = 'Floquet exponents'
    return _complex_make_axes(title, fig=fig, prop_cycle=prop_cycle)


def exponents(exps, ax=None, fig=None, legend=True, **kwds):
    if ax is None:
        ax = exponents_make_axes(fig=fig)
    return _complex(exps, ax, legend, **kwds)
