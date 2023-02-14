#!/bin/python

import numpy
import scipy.optimize
import scipy.sparse

from context import models
import models.birth
import models.death
import models.parameters
from models import _utility


class BirthConstant(models.birth.BirthConstant):
    def __init__(self, parameters):
        self.variation = parameters.birth_variation
        self.period = parameters.birth_period
        self.age_menarche = parameters.birth_age_menarche
        self.age_menopause = parameters.birth_age_menopause
        self.mean = 0.5


class BirthPeriodic(models.birth.BirthPeriodic):
    def __init__(self, parameters):
        self.variation = parameters.birth_variation
        self.period = parameters.birth_period
        self.age_menarche = parameters.birth_age_menarche
        self.age_menopause = parameters.birth_age_menopause
        self.mean = 0.5


class _Solver:
    '''Solver for the monodromy matrix of a linear age-structured
    model for the population size with age-dependent death rate,
    age-dependent maternity, and periodic time-dependent birth rate.'''

    def __init__(self, birth, death,
                 age_step=0.1, age_max=50,
                 birth_mid=True):
        self.birth = birth
        self.death = death
        self.age_step = self.time_step = age_step
        self.birth_mid = birth_mid
        self.period = self.birth.period
        if self.period == 0:
            self.period = self.time_step
        self.ages = _utility.build_t(0, age_max, self.age_step)
        self.times = _utility.build_t(0, self.period, self.time_step)
        assert numpy.isclose(self.times[-1], self.period)
        self._sol_cur = numpy.empty((len(self.ages), ) * 2)
        self._sol_new = numpy.empty((len(self.ages), ) * 2)
        self._init_matrices()

    def _init_matrices(self):
        '''Build the Crank–Nicolson and birth matrices.'''
        J = len(self.ages)
        H0 = _utility.sparse.diags({0: numpy.ones(J)})
        H1 = _utility.sparse.diags({-1: numpy.ones(J - 1),
                                    0: numpy.hstack([numpy.zeros(J - 1), 1])})
        mu = self.death.rate(self.ages)
        F0 = _utility.sparse.diags({0: -mu})
        F1 = _utility.sparse.diags({-1: -mu[:-1],
                                    0: numpy.hstack([numpy.zeros(J - 1), -mu[-1]])})
        self._HF0 = scipy.sparse.csr_array(H0 - self.time_step / 2 * F0)
        self._HF1 = scipy.sparse.csr_array(H1 + self.time_step / 2 * F1)
        nu = self.birth.maternity(self.ages)
        B = scipy.sparse.lil_array((J, J))
        B[0] = nu * self.age_step / 2
        self._B = scipy.sparse.csr_array(B)

    def _set_initial_condition(self):
        '''Set the initial condition.'''
        self._sol_new[:] = numpy.eye(len(self.ages))

    def _step(self, t_cur, birth_scaling):
        '''Do a step of the solver.'''
        # Update so that what was the new value of the solution is now
        # the current value and what was the current value of the
        # solution will be storage space for the new value.
        (self._sol_cur, self._sol_new) = (self._sol_new, self._sol_cur)
        if self.birth_mid:
            t_mid = t_cur + self.time_step / 2
            B0 = B1 = (birth_scaling
                       * self.birth.rate(t_mid)
                       * self._B)
        else:
            t_new = t_cur + self.time_step
            B0 = (birth_scaling
                  * self.birth.rate(t_new)
                  * self._B)
            B1 = (birth_scaling
                  * self.birth.rate(t_cur)
                  * self._B)
        HFB0 = self._HF0 - B0
        HFB1 = self._HF1 + B1
        HFBphi1 = HFB1 @ self._sol_cur
        self._sol_new[:] = scipy.sparse.linalg.spsolve(HFB0, HFBphi1)

    def solve_monodromy(self, birth_scaling=1):
        '''Get the monodromy matrix Psi = Phi(T), where Phi is the
        fundmental solution and T is the period.'''
        if len(self.times) == 0:
            return None
        self._set_initial_condition()
        for t_cur in self.times[:-1]:
            self._step(t_cur, birth_scaling)
        return self._sol_new

    def population_growth_rate(self, birth_scaling):
        '''Get the population growth rate.'''
        monodromy = self.solve_monodromy(birth_scaling)
        # Get the dominant Floquet multiplier.
        rho0 = _utility.get_dominant_eigen(monodromy, which='LM',
                                           return_eigenvector=False)
        # Convert the dominant Floquet multiplier to
        # the dominant Floquet exponent.
        mu0 = numpy.log(rho0) / self.period
        return mu0


def birth_scaling_for_zero_population_growth(Birth, *args, **kwds):
    '''Find the birth scaling that gives zero population growth rate.'''
    parameters = models.parameters.Parameters()
    birth = Birth(parameters)
    death = models.death.Death(parameters)
    solver = _Solver(birth, death, *args, **kwds)
    # `_Solver.population_growth_rate()` is increasing in
    # `birth_scaling`. Find a starting bracket `(lower, upper)` with
    # `solver.population_growth_rate(upper) > 0` and
    # `solver.population_growth_rate(lower) < 0`.
    MULT = 2
    upper = 1  # Starting guess.
    while solver.population_growth_rate(upper) < 0:
        upper *= MULT
    lower = upper / MULT  # Starting guess.
    while solver.population_growth_rate(lower) > 0:
        (lower, upper) = (lower / MULT, lower)
    return scipy.optimize.brentq(solver.population_growth_rate,
                                 lower, upper)


if __name__ == '__main__':
    scaling = birth_scaling_for_zero_population_growth(BirthPeriodic,
                                                       birth_mid=True)
    print(f'{scaling=}')
