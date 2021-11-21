#!/usr/bin/python3
'''Simple ODE test. Based on our FMDV work, this is a
non-age-structured SIR model with periodic birth rate.'''

import dataclasses

import matplotlib.pyplot
import numpy

import solver
import utility


class Solution:
    '''Model solution.'''
    def __init__(self, t, y, states):
        self.t = t
        self.y = y
        self.states = states

    def distance(self, t_0, t_1):
        '''Distance between solutions at `time_0` and `time_1`.'''
        y = utility.interp([t_0, t_1], self.t, self.y)
        return numpy.linalg.norm(y[..., 0] - y[..., 1])

    def is_periodic(self, period, tol=1e-8):
        '''Whether the tail of the solution is periodic with period
        `period`.'''
        t_1 = solution.t[-1]
        t_0 = t_1 - period
        assert t_0 >= solution.t[0], \
            f'{t_0=} is outside of the solution domain!'
        return self.distance(t_0, t_1) < tol

    def plot(self, axes=None, fig=None, legend=True, show=True, **kwds):
        '''Plot the solution.'''
        if axes is None:
            if fig is None:
                fig = matplotlib.pyplot.figure()
            axes = fig.add_subplot()
        axes.plot(self.t, self.y, **kwds)
        axes.set_xlabel('time')
        axes.set_ylabel('number')
        if legend:
            axes.legend(self.states)
        if show:
            matplotlib.pyplot.show()
        return axes

    def plot_population_size(self, axes=None, fig=None, show=True):
        '''Plot the population size.'''
        if axes is None:
            if fig is None:
                fig = matplotlib.pyplot.figure()
            axes = fig.add_subplot()
        axes.plot(self.t, self.y.sum(axis=1))
        axes.set_xlabel('time')
        axes.set_ylabel('population size')
        if show:
            matplotlib.pyplot.show()
        return axes


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

    def birth_rate(self, time):
        '''Periodic birth rate.'''
        return (self.parameters.death_rate
                * (1 + (self.parameters.birth_rate_variation
                        * numpy.sqrt(2)
                        * numpy.cos(2 * numpy.pi * time))))

    def rhs(self, time, state):
        '''The right-hand-side of the model ODEs.'''
        (susceptible, infectious, recovered) = state
        population_size = state.sum(axis=0)
        dsusceptible = (self.birth_rate(time) * population_size
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


class ModelConstantBirth(Model):
    '''The SIR model with constant birth rate.'''

    def __init__(self, **kwds):
        super().__init__(birth_rate_variation=0, **kwds)


if __name__ == '__main__':
    (t_start, t_end, t_step) = (0, 10, 0.001)
    model = Model()
    solution = model.solve(t_start, t_end, t_step)
    axes = solution.plot(linestyle='solid', show=False)
    model_constant = ModelConstantBirth()
    solution_constant = model_constant.solve(t_start, t_end, t_step)
    axes.set_prop_cycle(None)  # Reset color cycle
    solution_constant.plot(linestyle='dotted', legend=False, axes=axes)
