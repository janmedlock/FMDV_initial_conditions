'''Based on our FMDV work, this is an unstructured model.'''

import scipy.integrate

from . import _equilibrium
from . import _limit_cycle
from . import _solver
from .. import death
from .. import model
from .. import _utility


class Model(model.Base):
    '''Unstructured model.'''

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.death_rate_mean = death.rate_population_mean(self.birth_rate)
        self.birth_rate.mean \
            = self._birth_rate_mean_for_zero_population_growth()

    def _birth_rate_mean_for_zero_population_growth(self):
        '''For this unstructured model, the mean population growth
        rate is `self.birth_rate.mean - self.death_rate_mean`.'''
        return self.death_rate_mean

    def __call__(self, t, y):
        '''The right-hand-side of the model ODEs.'''
        (M, S, E, I, R) = y
        N = y.sum(axis=0)
        dM = (
            self.birth_rate(t) * N
            - 1 / self.parameters.maternal_immunity_mean * M
            - self.death_rate_mean * M
        )
        dS = (
            1 / self.parameters.maternal_immunity_mean * M
            - self.parameters.transmission_rate * I * S
            - self.death_rate_mean * S
        )
        dE = (
            self.parameters.transmission_rate * I * S
            - 1 / self.parameters.progression_mean * E
            - self.death_rate_mean * E
        )
        dI = (
            1 / self.parameters.progression_mean * E
            - 1 / self.parameters.recovery_mean * I
            - self.death_rate_mean * I
        )
        dR = (
            1 / self.parameters.recovery_mean * I
            - self.death_rate_mean * R
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
        t = _utility.arange(t_start, t_end, t_step)
        if y_start is None:
            y_start = self.build_initial_conditions()
        sol = _solver.solve(self, t, y_start,
                            states=self.states)
        _utility.assert_nonnegative(sol)
        return sol

    def find_equilibrium(self, y_0_guess):
        '''Find an equilibrium of the model.'''
        eql = _equilibrium.find(self, 0, y_0_guess,
                                states=self.states)
        _utility.assert_nonnegative(eql)
        return eql

    def get_eigenvalues(self, eql):
        '''Get the eigenvalues of the Jacobian.'''
        return _equilibrium.eigenvalues(self, 0, eql)

    def find_limit_cycle(self, t_0, period, t_step, y_0_guess):
        '''Find a limit cycle of the model.'''
        lcy = _limit_cycle.find(self, t_0, period, t_step,
                                y_0_guess,
                                states=self.states)
        _utility.assert_nonnegative(lcy)
        return lcy

    def get_characteristic_multipliers(self, lcy):
        '''Get the characteristic multipliers.'''
        return _limit_cycle.characteristic_multipliers(self, lcy)

    def get_characteristic_exponents(self, lcy):
        '''Get the characteristic exponents.'''
        return _limit_cycle.characteristic_exponents(self, lcy)
