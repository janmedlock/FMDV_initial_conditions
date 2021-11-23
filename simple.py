#!/usr/bin/python3
'''Simple ODE test. Based on our FMDV work, this is a
non-age-structured SIR model with periodic birth rate.'''

import dataclasses

import matplotlib.pyplot
import numpy
import pandas
import scipy.optimize

import solver
import utility


class _StateBase:
    def __init__(self, data):
        self._data = data

    @staticmethod
    def _index_states(states):
        return pandas.Index(states, name='state')

    def __repr__(self):
        return self._data.__repr__()

    @property
    def y(self):
        '''State values.'''
        return self._data.to_numpy()

    @property
    def states(self):
        '''State names.'''
        return self._data.axes[-1]

    @property
    def population_size(self):
        '''Population size.'''
        axis = self._data.ndim - 1  # Sum over last axis.
        return self._data.sum(axis=axis)

    def _get_axes(self):
        if len(self.states) == 2:
            projection = 'rectilinear'
        elif len(self.states) == 3:
            projection = '3d'
        else:
            raise ValueError(
                f'State dimension is {len(self._data.states)}, '
                'but only 2 and 3 are supported!')
        return matplotlib.pyplot.figure().add_subplot(projection=projection)

    def _label_axes(self, ax):
        ax.set_xlabel(self.states[0])
        if len(self.states) >= 2:
            ax.set_ylabel(self.states[1])
        if len(self.states) >= 3:
            ax.set_zlabel(self.states[2])


class State(_StateBase):
    '''Model state coordinates.'''
    def __init__(self, y, states):
        data = pandas.Series(y,
                             index=self._index_states(states))
        super().__init__(data)

    def plot(self, ax=None, label_axes=True, **kwds):
        '''Plot the point in state space.'''
        if ax is None:
            ax = self._get_axes()
        ax.scatter(*self.y)
        if label_axes:
            self._label_axes(ax)
        return ax


class Solution(_StateBase):
    '''Model solution.'''
    def __init__(self, t, y, states):
        data = pandas.DataFrame(y,
                                index=self._index_t(t),
                                columns=self._index_states(states))
        super().__init__(data)

    @staticmethod
    def _index_t(t):
        return pandas.Index(t, name='$t$')

    @property
    def t(self):
        '''Time values.'''
        return self._data.index.to_numpy()

    def interp(self, t):
        '''Interpolate to `t`.'''
        return utility.interp(t, self.t, self.y)

    def distance(self, t_0, t_1):
        '''Distance between solutions at `time_0` and `time_1`.'''
        (y_0, y_1) = self.interp([t_0, t_1])
        return numpy.linalg.norm(y_0 - y_1)

    def is_periodic(self, period, tol=1e-8):
        '''Whether the tail of the solution is periodic with period
        `period`.'''
        t_1 = self.t[-1]
        t_0 = t_1 - period
        return self.distance(t_0, t_1) < tol

    def plot(self, **kwds):
        '''Plot the solution.'''
        return self._data.plot(**kwds)

    def plot_phase(self, ax=None, label_axes=True, **kwds):
        '''Make a phase plot.'''
        if ax is None:
            ax = self._get_axes()
        ax.plot(*self.y.T)
        if label_axes:
            self._label_axes(ax)
        return ax


@dataclasses.dataclass
class Parameters:
    '''Model parameters.'''
    death_rate: float = 0.1  # per year
    birth_rate_variation: float = 0.5
    transmission_rate: float = 2.8 * 365  # per year
    recovery_rate: float = 1 / 5.7 * 365  # per yera


class Model:
    '''The SIR model.'''
    STATES = ('susceptible', 'infectious', 'recovered')

    def __init__(self, **kwds):
        self.parameters = Parameters(**kwds)

    def birth_rate(self, t):
        '''Periodic birth rate.'''
        return (self.parameters.death_rate
                * (1 + (self.parameters.birth_rate_variation
                        * numpy.sqrt(2)
                        * numpy.cos(2 * numpy.pi * t))))

    def rhs(self, t, y):
        '''The right-hand-side of the model ODEs.'''
        (susceptible, infectious, recovered) = y
        population_size = y.sum(axis=0)
        dsusceptible = (self.birth_rate(t) * population_size
                        - (self.parameters.transmission_rate
                           * infectious * susceptible)
                        - self.parameters.death_rate * susceptible)
        dinfectious = ((self.parameters.transmission_rate
                        * infectious * susceptible)
                       - self.parameters.recovery_rate * infectious
                       - self.parameters.death_rate * infectious)
        drecovered = (self.parameters.recovery_rate * infectious
                      - self.parameters.death_rate * recovered)
        return (dsusceptible, dinfectious, drecovered)

    @staticmethod
    def build_initial_conditions():
        '''Build the initial conditions.'''
        infectious = 0.01
        recovered = 0
        susceptible = 1 - infectious - recovered
        return (susceptible, infectious, recovered)

    def solve(self, t_start, t_end, t_step, y_start=None):
        '''Solve the ODEs.'''
        t = utility.arange(t_start, t_end, t_step)
        if y_start is None:
            y_start = self.build_initial_conditions()
        y = solver.solve(self.rhs, t, y_start)
        assert (y >= 0).all()
        return Solution(t, y, self.STATES)

    def _objective_equilibrium(self, y, t):
        return self.rhs(t, y)

    def find_equilibrium(self, t, y_0):
        '''Find an equilibrium of the model.'''
        result = scipy.optimize.root(self._objective_equilibrium,
                                     y_0, args=(t, ))
        assert result.success, f'{result}'
        return State(result.x, self.STATES)

    def _objective_limit_cycle(self, y_0, solver_, t, y):
        y[0] = y_0
        solver_.solve(t, y=y)
        return y[-1] - y[0]

    def find_limit_cycle(self, t_start, period, t_step, y_start):
        '''Find a limit cycle of the model.'''
        t = utility.arange(t_start, t_start + period, t_step)
        y = numpy.empty((len(t), *numpy.shape(y_start)))
        solver_ = solver.Solver.create(self.rhs)
        result = scipy.optimize.root(self._objective_limit_cycle,
                                     y_start, args=(solver_, t, y))
        assert result.success, f'{result}'
        y[0] = result.x
        solver_.solve(t, y=y)
        return Solution(t, y, self.STATES)


class ModelConstantBirth(Model):
    '''The SIR model with constant birth rate.'''

    def __init__(self, **kwds):
        super().__init__(birth_rate_variation=0, **kwds)


if __name__ == '__main__':
    (t_start, t_end, t_step) = (0, 10, 0.001)
    model_constant = ModelConstantBirth()
    solution_constant = model_constant.solve(t_start, t_end, t_step)
    ax = solution_constant.plot(linestyle='dotted', legend=False)
    ax.set_prop_cycle(None)  # Reset color cycle
    equilibrium = model_constant.find_equilibrium(t_end,
                                                  solution_constant.y[-1])
    model = Model()
    solution = model.solve(t_start, t_end, t_step)
    solution.plot(ax=ax)
    period = 1
    limit_cycle = model.find_limit_cycle(t_end, period, t_step,
                                         solution.y[-1])
    matplotlib.pyplot.show()
