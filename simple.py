#!/usr/bin/python3
'''Simple ODE test. Based on our FMDV work, this is a
non-age-structured SIR model with periodic birth rate.'''

import dataclasses

import matplotlib.pyplot
import numpy

import solver


@dataclasses.dataclass
class Parameters:
    '''Model parameters.'''
    death_rate: float = 0.1  # per year
    birth_rate_variation: float = 0.5
    transmission_rate: float = 2.8 * 365  # per year
    recovery_rate: float = 1 / 5.7 * 365  # per year


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

    def __call__(self, t, y):
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

    @staticmethod
    def assert_nonnegative_solution(sol):
        '''Check the that `sol` is non-negative.'''
        assert (sol.y >= 0).all()

    def solve(self, t_start, t_end, t_step, y_start=None):
        '''Solve the ODEs.'''
        t = solver.utility.arange(t_start, t_end, t_step)
        if y_start is None:
            y_start = self.build_initial_conditions()
        sol = solver.solve(self, t, y_start, states=self.STATES)
        self.assert_nonnegative_solution(sol)
        return sol

    def find_equilibrium(self, t, y_0_guess):
        '''Find an equilibrium of the model.'''
        eql = solver.equilibrium.find(self, t, y_0_guess, states=self.STATES)
        self.assert_nonnegative_solution(eql)
        return eql

    def find_limit_cycle(self, t_0, period, t_step, y_0_guess):
        '''Find a limit cycle of the model.'''
        lcy = solver.limit_cycle.find(self, t_0, period, t_step,
                                      y_0_guess, states=self.STATES)
        self.assert_nonnegative_solution(lcy)
        return lcy


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
    ax_phase = equilibrium.plot()
    model = Model()
    solution = model.solve(t_start, t_end, t_step)
    solution.plot(ax=ax)
    PERIOD = 1
    limit_cycle = model.find_limit_cycle(t_end, PERIOD, t_step, solution.y[-1])
    limit_cycle.plot_phase(ax=ax_phase)
    matplotlib.pyplot.show()
