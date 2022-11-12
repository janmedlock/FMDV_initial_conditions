'''Based on our FMDV work, this is an unstructured model.'''

from . import equilibrium
from . import limit_cycle
from . import parameters
from . import solver
from .. import model
from .. import utility


class _Model(model.Model):
    '''Base class for unstructured models.'''

    def __call__(self, t, y):
        '''The right-hand-side of the model ODEs.'''
        (M, S, E, I, R) = y
        N = y.sum(axis=0)
        dM = (
            self.parameters.birth_rate(t) * N
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
        t = utility.arange(t_start, t_end, t_step)
        if y_start is None:
            y_start = self.build_initial_conditions()
        sol = solver.solve(self, t, y_start,
                           states=self.states)
        utility.assert_nonnegative(sol)
        return sol


class ModelBirthPeriodic(_Model):
    '''Unstructured model with periodic birth rate.'''

    _Parameters = parameters.ParametersBirthPeriodic

    def find_limit_cycle(self, t_0, period, t_step, y_0_guess):
        '''Find a limit cycle of the model.'''
        lcy = limit_cycle.find(self, t_0, period, t_step,
                               y_0_guess,
                               states=self.states)
        utility.assert_nonnegative(lcy)
        return lcy

    def get_characteristic_multipliers(self, lcy):
        '''Get the characteristic multipliers.'''
        return limit_cycle.characteristic_multipliers(self, lcy)

    def get_characteristic_exponents(self, lcy):
        '''Get the characteristic exponents.'''
        return limit_cycle.characteristic_exponents(self, lcy)


class ModelBirthConstant(_Model):
    '''Unstructured model with constant birth rate.'''

    _Parameters = parameters.ParametersBirthConstant

    def find_equilibrium(self, y_0_guess):
        '''Find an equilibrium of the model.'''
        eql = equilibrium.find(self, 0, y_0_guess,
                               states=self.states)
        utility.assert_nonnegative(eql)
        return eql

    def get_eigenvalues(self, eql):
        '''Get the eigenvalues of the Jacobian.'''
        return equilibrium.eigenvalues(self, 0, eql)
