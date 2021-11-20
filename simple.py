#!/usr/bin/python3
'''Simple ODE test. Based on our FMDV work, this is a
non-age-structured SIR model with periodic birth rate.'''

import dataclasses

import matplotlib.pyplot
import numpy
import scipy.special

import solver
import utility


class Solution:
    '''Model solution.'''
    def __init__(self, t, y, log, states):
        self.t = t
        self._y = y
        self.log = log
        self.states = states

    @property
    def y(self):
        '''The solution, untransformed, if necessary.'''
        y = self._y
        if self.log:
            y = numpy.exp(y)
        return y

    def interp(self, t):
        '''Interpolate the solution.'''
        # Interpolate, then untransform, if necessary.
        y = utility.interp(t, self.t, self._y)
        if self.log:
            y = numpy.exp(y)
        return y

    def distance(self, t_0, t_1):
        '''Distance between solutions at `time_0` and `time_1`.'''
        y = self.interp([t_0, t_1])
        return numpy.linalg.norm(y[..., 0] - y[..., 1])

    def is_periodic(self, period, tol=1e-8):
        '''Whether the tail of the solution is periodic with period
        `period`.'''
        t_1 = solution.t[-1]
        t_0 = t_1 - period
        assert t_0 >= solution.t[0], \
            f'{t_0=} is outside of the solution domain!'
        return self.distance(t_0, t_1) < tol

    def plot(self, show=True):
        '''Plot the solution.'''
        (fig, axes) = matplotlib.pyplot.subplots()
        axes.plot(self.t, self.y)
        axes.set_xlabel('time')
        axes.set_ylabel('number')
        axes.legend(self.states)
        if show:
            matplotlib.pyplot.show()
        return fig

    def plot_population_size(self, show=True):
        '''Plot the population size.'''
        (fig, axes) = matplotlib.pyplot.subplots()
        axes.plot(self.t, self.y.sum(axis=1))
        axes.set_xlabel('time')
        axes.set_ylabel('population size')
        if show:
            matplotlib.pyplot.show()
        return fig


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

    def __init__(self, *args, **kwds):
        self.parameters = Parameters(*args, **kwds)

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
        return numpy.array((dsusceptible, dinfectious, drecovered))

    def rhs_log(self, time, state_log):
        '''The right-hand-side of the model ODEs for the log-transformed state
        variables.'''
        (susceptible_log, infectious_log, recovered_log) = state_log
        population_size_log = scipy.special.logsumexp(state_log, axis=0)
        dsusceptible_log = ((self.birth_rate(time)
                             * numpy.exp(population_size_log
                                         - susceptible_log))
                            - (self.parameters.transmission_rate
                               * numpy.exp(infectious_log))
                            - self.parameters.death_rate)
        dinfectious_log = ((self.parameters.transmission_rate
                            * numpy.exp(susceptible_log))
                           - self.parameters.recovery_rate
                           - self.parameters.death_rate)
        drecovered_log = ((self.parameters.recovery_rate
                           * numpy.exp(infectious_log
                                       - recovered_log))
                          - self.parameters.death_rate)
        return numpy.array((dsusceptible_log, dinfectious_log, drecovered_log))

    @staticmethod
    def build_initial_conditions():
        '''Build the initial conditions.'''
        infectious = 0.01
        recovered = 0
        susceptible = 1 - infectious - recovered
        return numpy.array((susceptible, infectious, recovered))

    def solve(self, t_start, t_end, t_step, y_0=None, log=False,
              _log_of_zero=-20):
        '''Solve the ODEs.'''
        if y_0 is None:
            y_0 = self.build_initial_conditions()
        if not log:
            func = self.rhs
        else:
            func = self.rhs_log
            assert numpy.all(y_0 >= 0)
            y_0 = numpy.ma.filled(numpy.ma.log(y_0), _log_of_zero)
        (t, y) = solver.solver(func, t_start, t_end, t_step, y_0)
        if not log:
            assert (y >= 0).all()
        return Solution(t, y, log, self.STATES)


if __name__ == '__main__':
    model = Model()
    solution = model.solve(0, 10, 0.001)
    figure = solution.plot()
