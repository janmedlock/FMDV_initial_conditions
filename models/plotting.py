'''Plot solutions and points in state space.'''

import functools

import matplotlib.pyplot
import matplotlib.rcsetup


_linestyles = matplotlib.rcsetup.cycler(
    linestyle=['solid', 'dotted', 'dashed']
)

_markers = matplotlib.rcsetup.cycler(
    marker=['o', '^', 's', '*', '+', 'x']
)


def _cycler_inner_product(*cyclers):
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


def _state_prop_cycle():
    '''Build a prop_cycle for the state plot.'''
    orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
    prop_cycle = _cycler_inner_product(orig, _markers, _linestyles)
    return prop_cycle


def _state_make_axes(states):
    '''Make state axes.'''
    if len(states) == 2:
        projection = 'rectilinear'
    elif len(states) == 3:
        projection = '3d'
    else:
        raise ValueError(f'{len(states)=}')
    fig = matplotlib.pyplot.figure()
    axis_labels = dict(zip(('xlabel', 'ylabel', 'zlabel'), states))
    return fig.add_subplot(projection=projection,
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


def state(obj, states=None, ax=None, **kwds):
    '''Make a phase plot.'''
    if ax is None:
        ax = _state_make_axes(_get_states(states, obj))
    if states is None:
        states = _get_states_from_axes(ax)
    if obj.ndim == 1:
        ax.scatter(*obj[states], **kwds)
    elif obj.ndim == 2:
        # Not `col = obj[states]` because the result must iterate over
        # columns, not rows.
        cols = (obj[col] for col in states)
        ax.plot(*cols, **kwds)
    else:
        raise ValueError(f'{obj.ndim=}')
    return ax


def _solution_prop_cycle(states):
    '''Build a prop_cycle for the solution plot.'''
    orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
    inner = orig[:len(states)]
    outer = _linestyles
    prop_cycle = outer * inner
    return prop_cycle


def _solution_make_axes(states):
    '''Make solution axes.'''
    fig = matplotlib.pyplot.figure()
    return fig.add_subplot(
        prop_cycle=_solution_prop_cycle(states),
    )


def solution(obj, states=None, ax=None, **kwds):
    '''Plot the solution vs time.'''
    states = _get_states(states, obj)
    if ax is None:
        ax = _solution_make_axes(states)
    return obj[states].plot(ax=ax, **kwds)


def _complex_prop_cycle():
    '''Build a prop_cycle for the eigenvalue plot.'''
    orig = matplotlib.pyplot.rcParams['axes.prop_cycle']
    prop_cycle = _cycler_inner_product(orig, _markers)
    return prop_cycle


def _complex_make_axes(title):
    fig = matplotlib.pyplot.figure()
    return fig.add_subplot(
        title=title,
        xlabel='$\\Re$',
        ylabel='$\\Im$',
        prop_cycle=_complex_prop_cycle(),
    )


def _complex(vals, ax, legend, **kwds):
    ax.plot(vals.real, vals.imag, linestyle='', **kwds)
    if legend:
        ax.legend()
    return ax


def multipliers(mults, ax=None, legend=False, **kwds):
    if ax is None:
        ax = _complex_make_axes('Floquet multipliers')
    return _complex(mults, ax, legend, **kwds)


def exponents(exps, ax=None, legend=False, **kwds):
    if ax is None:
        ax = _complex_make_axes('Floquent exponents')
    return _complex(exps, ax, legend, **kwds)
