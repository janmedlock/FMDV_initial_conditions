#!/usr/bin/python3
'''Based on our FMDV work, this is an unstructured model with
periodic birth rate.'''

import dataclasses

import matplotlib.pyplot
import numpy

import birth
import solvers
import solvers.unstructured


@dataclasses.dataclass
class Parameters:
    '''Model parameters.'''
    death_rate: float = 0.1  # per year
    birth_rate_variation: float = 0.5
    maternal_immunity_waning_rate: float =  1 / 0.37  # per year
    transmission_rate: float = 2.8 * 365  # per year
    progression_rate = 1 / 0.5 * 365  # per year
    recovery_rate: float = 1 / 5.7 * 365  # per year


class Model(birth.PeriodicBirthRateMixin):
    '''Unstructured model with periodic birth rate.'''

    STATES = ('maternal_immunity', 'susceptible', 'exposed',
              'infectious', 'recovered')

    PERIOD = 1  # year

    def __init__(self, **kwds):
        self.parameters = Parameters(**kwds)
        self.birth_rate_mean = self.parameters.death_rate

    def __call__(self, t, y):
        '''The right-hand-side of the model ODEs.'''
        (M, S, E, I, R) = y
        N = y.sum(axis=0)
        dM = (
            self.birth_rate(t) * N
            - self.parameters.maternal_immunity_waning_rate * M
            - self.parameters.death_rate * M
        )
        dS = (
            self.parameters.maternal_immunity_waning_rate * M
            - self.parameters.transmission_rate * I * S
            - self.parameters.death_rate * S
        )
        dE = (
            self.parameters.transmission_rate * I * S
            - self.parameters.progression_rate * E
            - self.parameters.death_rate * E
        )
        dI = (
            self.parameters.progression_rate * E
            - self.parameters.recovery_rate * I
            - self.parameters.death_rate * I
        )
        dR = (
            self.parameters.recovery_rate * I
            - self.parameters.death_rate * R
        )
        return (dM, dS, dE, dI, dR)

    @staticmethod
    def build_initial_conditions():
        '''Build the initial conditions.'''
        M = 0
        E = 0
        I = 0.01
        R = 0
        S = 1 - M - E - I - R
        return (M, S, E, I, R)

    def solve(self, t_start, t_end, t_step, y_start=None):
        '''Solve the ODEs.'''
        t = solvers.utility.arange(t_start, t_end, t_step)
        if y_start is None:
            y_start = self.build_initial_conditions()
        sol = solvers.unstructured.solve(self, t, y_start,
                                         states=self.STATES)
        solvers.utility.assert_nonnegative(sol)
        return sol

    def find_limit_cycle(self, t_0, period, t_step, y_0_guess):
        '''Find a limit cycle of the model.'''
        lcy = solvers.unstructured.limit_cycle.find(self, t_0, period, t_step,
                                                    y_0_guess,
                                                    states=self.STATES)
        solvers.utility.assert_nonnegative(lcy)
        return lcy

    def get_characteristic_exponents(self, lcy):
        '''Get the characteristic exponents.'''
        return solvers.unstructured.limit_cycle.characteristic_exponents(self,
                                                                         lcy)


class ModelConstantBirth(Model):
    '''The SIR model with constant birth rate.'''

    def __init__(self, **kwds):
        super().__init__(birth_rate_variation=0, **kwds)

    def find_equilibrium(self, y_0_guess):
        '''Find an equilibrium of the model.'''
        eql = solvers.unstructured.equilibrium.find(self, 0, y_0_guess,
                                                    states=self.STATES)
        solvers.utility.assert_nonnegative(eql)
        return eql

    def get_eigenvalues(self, eql):
        '''Get the eigenvalues of the Jacobian.'''
        return solvers.unstructured.equilibrium.eigenvalues(self,
                                                            0, eql)


if __name__ == '__main__':
    (t_start, t_end, t_step) = (0, 10, 0.001)
    model = Model()
    solution = model.solve(t_start, t_end, t_step)
    ax_solution = solution.solution.plot_solution()
    limit_cycle = model.find_limit_cycle(t_end, model.PERIOD, t_step,
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
