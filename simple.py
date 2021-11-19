#!/usr/bin/python3
'''Simple ODE test. Based on our FMDV work, this is a
non-age-structured SIR model with periodic birth rate.'''

import dataclasses
import math

import matplotlib.pyplot
import numpy
import scipy.integrate
import scipy.special


def arange(start, stop, step, endpoint=True, dtype=None):
    '''Like `numpy.arange()` but ensure that
    * `stop - step` is an integer multiple of `step`, and
    * last point is in the output if `endpoint` is True.'''
    # Make `stop - start` an integer multiple of `step`.
    num = math.ceil((stop - start) / step)
    stop = start + num * step
    if endpoint:
        num += 1
    return numpy.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)


class Solution:
    '''Model solution.'''
    def __init__(self, solution, states, log):
        self.solution = solution
        self.states = states
        self.log = log

    @property
    def t(self):
        '''The times.'''
        return self.solution.t

    @property
    def y(self):
        '''The solution.'''
        val = self.solution.y
        if self.log:
            val = numpy.exp(val)
        return val

    def interp(self, time):
        '''Interpolate the solution.'''
        val = numpy.interp(time, self.solution.t, self.solution.y)
        if self.log:
            val = numpy.exp(val)
        return val

    def distance(self, time_0, time_1, ord=2):
        '''Distance between solutions at `time_0` and `time_1`.'''
        sols = self.interp([time_0, time_1])
        return numpy.linalg.norm(sols[:, 0] - sols[:, 1], ord=ord)

    def is_periodic(self, period, ord=None, tol=1e-8):
        '''Whether the solution periodic with period `period`.'''
        time_1 = solution.t[-1]
        time_0 = time_1 - period
        assert time_0 >= solution.t[0]
        distance = self.distance(time_0, time_1, ord=ord)
        print(distance)
        return distance < tol

    def plot(self, show=True):
        '''Plot the solution.'''
        (figure, axes) = matplotlib.pyplot.subplots()
        axes.plot(self.t, self.y.T)
        axes.set_xlabel('time')
        axes.set_ylabel('number')
        axes.legend(self.states)
        if show:
            matplotlib.pyplot.show()
        return figure

    def plot_population_size(self, points=301, show=True):
        '''Plot the population size.'''
        (figure, axes) = matplotlib.pyplot.subplots()
        time = numpy.linspace(*self.t[[0, -1]], points)
        axes.plot(time, self.sol(time).sum(0))
        axes.set_xlabel('time')
        axes.set_ylabel('population size')
        if show:
            matplotlib.pyplot.show()
        return figure


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
        return (dsusceptible, dinfectious, drecovered)

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
        return (dsusceptible_log, dinfectious_log, drecovered_log)

    @staticmethod
    def build_initial_conditions():
        '''Build the initial conditions.'''
        infectious = 0.01
        recovered = 0
        susceptible = 1 - infectious - recovered
        return numpy.array((susceptible, infectious, recovered))

    def solve(self, time_start, time_end, time_step,
              initial_conditions=None, log=True, _log_of_zero=-20,
              **kwds):
        '''Solve the ODEs.'''
        if initial_conditions is None:
            initial_conditions = self.build_initial_conditions()
        if not log:
            func = self.rhs
        else:
            func = self.rhs_log
            initial_conditions = numpy.ma.log(initial_conditions).filled(
                _log_of_zero)
        times = arange(time_start, time_end, time_step)
        sol = scipy.integrate.solve_ivp(func,
                                        (time_start, time_end),
                                        initial_conditions,
                                        t_eval=times,
                                        **kwds)
        return Solution(sol, self.STATES, log)


if __name__ == '__main__':
    model = Model()
    solution = model.solve(0, 10, 0.01)
    figure = solution.plot()
