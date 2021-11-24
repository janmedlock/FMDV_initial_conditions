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
        d_susceptible = (
            self.birth_rate(t) * population_size
            - self.parameters.transmission_rate * infectious * susceptible
            - self.parameters.death_rate * susceptible
        )
        d_infectious = (
            self.parameters.transmission_rate * infectious * susceptible
            - self.parameters.recovery_rate * infectious
            - self.parameters.death_rate * infectious
        )
        d_recovered = (
            self.parameters.recovery_rate * infectious
            - self.parameters.death_rate * recovered
        )
        return (d_susceptible, d_infectious, d_recovered)

    def jacobian(self, t, y):
        '''The Jacobian of the model ODEs.'''
        (susceptible, infectious, recovered) = y
        grad_susceptible = (
            (self.birth_rate(t)
             - self.parameters.transmission_rate * infectious
             - self.parameters.death_rate),
            (self.birth_rate(t)
             - self.parameters.transmission_rate * susceptible),
            self.birth_rate(t)
        )
        grad_infectious = (
            self.parameters.transmission_rate * infectious,
            (self.parameters.transmission_rate * susceptible
             - self.parameters.recovery_rate
             - self.parameters.death_rate),
            0
        )
        grad_recovered = (
            0,
            self.parameters.recovery_rate,
            - self.parameters.death_rate
        )
        return (grad_susceptible, grad_infectious, grad_recovered)

    @staticmethod
    def build_initial_conditions():
        '''Build the initial conditions.'''
        infectious = 0.01
        recovered = 0
        susceptible = 1 - infectious - recovered
        return (susceptible, infectious, recovered)

    @staticmethod
    def assert_nonnegative(y):
        '''Check that `y` is non-negative.'''
        assert (y >= 0).all(axis=None)

    def solve(self, t_start, t_end, t_step, y_start=None):
        '''Solve the ODEs.'''
        t = solver.utility.arange(t_start, t_end, t_step)
        if y_start is None:
            y_start = self.build_initial_conditions()
        sol = solver.solve(self, t, y_start, states=self.STATES)
        self.assert_nonnegative(sol)
        return sol

    def find_limit_cycle(self, t_0, period, t_step, y_0_guess):
        '''Find a limit cycle of the model.'''
        lcy = solver.limit_cycle.find(self, t_0, period, t_step,
                                      y_0_guess, states=self.STATES)
        self.assert_nonnegative(lcy)
        return lcy

    def get_characteristic_exponents(self, lcy):
        '''Get the characteristic exponents.'''
        return solver.limit_cycle.characteristic_exponents(self.jacobian, lcy)


class ModelConstantBirth(Model):
    '''The SIR model with constant birth rate.'''

    def __init__(self, **kwds):
        super().__init__(birth_rate_variation=0, **kwds)

    def find_equilibrium(self, y_0_guess):
        '''Find an equilibrium of the model.'''
        eql = solver.equilibrium.find(self, 0, y_0_guess,
                                      states=self.STATES)
        self.assert_nonnegative(eql)
        return eql

    def get_eigenvalues(self, eql):
        '''Get the eigenvalues of the Jacobian.'''
        return solver.equilibrium.eigenvalues(self.jacobian, 0, eql)


if __name__ == '__main__':
    (t_start, t_end, t_step) = (0, 10, 0.001)
    model = Model()
    solution = model.solve(t_start, t_end, t_step)
    ax_solution = solution.solution.plot_solution()
    PERIOD = 1
    limit_cycle = model.find_limit_cycle(t_end, PERIOD, t_step,
                                         solution.loc[t_end])
    print(model.get_characteristic_exponents(limit_cycle))
    ax_state = limit_cycle.solution.plot_state()
    model_constant = ModelConstantBirth()
    solution_constant = model_constant.solve(t_start, t_end, t_step)
    solution_constant.solution.plot_solution(ax=ax_solution, legend=False)
    equilibrium = model_constant.find_equilibrium(solution_constant.loc[t_end])
    print(model_constant.get_eigenvalues(equilibrium))
    equilibrium.solution.plot_state(ax=ax_state)
    matplotlib.pyplot.show()
