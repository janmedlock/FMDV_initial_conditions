'''Based on our FMDV work, this is an unstructured model.'''

from . import _equilibrium
from . import _limit_cycle
from . import _solver
from .. import model
from .. import _utility


class Model(model.Base):
    '''Unstructured model.'''

    def __init__(self, **kwds):
        super().__init__(**kwds)
        # Use `self.birth` with age-dependent `.mean` to find
        # `self.death_rate_mean`.
        self.death_rate_mean = self.death.rate_population_mean(self.birth)
        # Set `self.birth.mean` so the non-age-dependent model has
        # zero population growth rate.
        self.birth.mean = self._birth_rate_mean_for_zero_population_growth()

    def _birth_rate_mean_for_zero_population_growth(self):
        '''For this unstructured model, the mean population growth
        rate is `self.birth_rate.mean - self.death_rate_mean`.'''
        return self.death_rate_mean

    def __call__(self, t, y):
        '''The right-hand-side of the model ODEs.'''
        (M, S, E, I, R) = y
        birth_rate_t = self.birth.rate(t)
        N_antibodies = y[self._states_have_antibodies].sum(axis=0)
        N_no_antibodies = y[~self._states_have_antibodies].sum(axis=0)
        dM = (
            birth_rate_t * N_antibodies
            - 1 / self.waning.mean * M
            - self.death_rate_mean * M
        )
        dS = (
            birth_rate_t * N_no_antibodies
            + 1 / self.waning.mean * M
            - self.transmission.rate * I * S
            - self.death_rate_mean * S
        )
        dE = (
            self.transmission.rate * I * S
            - 1 / self.progression.mean * E
            - self.death_rate_mean * E
        )
        dI = (
            1 / self.progression.mean * E
            - 1 / self.recovery.mean * I
            - self.death_rate_mean * I
        )
        dR = (
            1 / self.recovery.mean * I
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
