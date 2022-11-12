'''Models of population size.'''

import numpy
import pandas
import scipy.integrate
import scipy.optimize
import scipy.sparse

from . import _sparse
from .. import death
from .. import maternity
from .. import _utility


class _Solver:
    '''Solver for the monodromy matrix of a linear age-structured
    model for the population size with age-dependent death and
    maternity rates and periodic time-dependent birth rate.'''

    def __init__(self, birth_rate, age_step=0.1, age_max=50):
        self.birth_rate = birth_rate
        self.age_step = self.time_step = age_step
        self.period = self.birth_rate.period
        if self.period == 0:
            self.period = self.time_step
        self.ages = _utility.arange(0, age_max, self.age_step,
                                    endpoint=True)
        self.times = _utility.arange(0, self.period, self.time_step,
                                     endpoint=True)
        self._sol_curr = numpy.empty((len(self.ages), ) * 2)
        self._sol_prev = numpy.empty((len(self.ages), ) * 2)
        self._init_crank_nicolson()
        self._init_birth()

    def _init_crank_nicolson(self):
        '''Build the matrix used for the Crank–Nicolson step.'''
        mat_cn = scipy.sparse.lil_matrix((len(self.ages), ) * 2)
        # Midpoints between adjacent ages.
        ages_mid = (self.ages[1:] + self.ages[:-1]) / 2
        k = death.rate(ages_mid) * self.time_step / 2
        # Set the first subdiagonal.
        mat_cn.setdiag((1 - k) / (1 + k), -1)
        # Keep the last age group from ageing out.
        k_last = death.rate(self.ages[-1]) * self.time_step / 2
        mat_cn[-1, -1] = (1 - k_last) / (1 + k_last)
        self._mat_cn = _sparse.csr_matrix(mat_cn)

    def _init_birth(self):
        '''Build the vector used for the integral step.'''
        self._vec_birth = maternity.rate(self.ages) * self.age_step
        self._vec_birth[[0, -1]] /= 2
        # Temporary storage for efficiency.
        self._vec_temp = numpy.empty(len(self.ages))

    def _set_initial_condition(self):
        '''Set the initial condition.'''
        # self._sol_curr[:] = numpy.eye(len(self.ages))
        # Avoid build a new matrix.
        self._sol_curr[:] = 0
        numpy.fill_diagonal(self._sol_curr, 1)

    def _step_crank_nicolson(self):
        '''Do the Crank–Nicolson step.'''
        # self._sol_curr = self._mat_cn @ self._sol_prev
        # Avoid building a new matrix.
        self._sol_curr[:] = 0
        # self._sol_curr += self._mat_cn @ self._sol_prev
        self._mat_cn.matvecs(self._sol_prev, self._sol_curr)

    def _step_birth(self, t_curr, birth_scaling):
        '''Do the birth step.'''
        # self._sol_curr[0] = (birth_scaling
        #                      * self.birth_rate(t_curr)
        #                      * self._vec_birth @ self._sol_curr)
        # Avoid building new vectors.
        self._vec_temp[:] = self._vec_birth
        self._vec_temp *= birth_scaling * self.birth_rate(t_curr)
        self._vec_temp.dot(self._sol_curr, out=self._sol_curr[0])

    def _step(self, t_curr, birth_scaling):
        '''Do a step of the solver.'''
        # Update so that what was the current value of the solulation
        # is now the previous value and what was the previous value of
        # the solulation will be storage space for the new current
        # value.
        (self._sol_curr, self._sol_prev) = (self._sol_prev, self._sol_curr)
        self._step_crank_nicolson()
        self._step_birth(t_curr, birth_scaling)

    def solve_monodromy(self, birth_scaling=1):
        '''Get the monodromy matrix Psi = Phi(T), where Phi is the
        fundmental solution and T is the period.'''
        if len(self.times) == 0:
            return None
        self._set_initial_condition()
        for t_curr in self.times[1:]:
            self._step(t_curr, birth_scaling)
        return self._sol_curr

    def get_pop_growth(self, birth_scaling):
        '''Get the population growth rate.'''
        monodromy = self.solve_monodromy(birth_scaling)
        # Get the dominant Floquet multiplier.
        rho0 = _utility.get_dominant_eigen(monodromy, which='LM',
                                           return_eigenvector=False)
        # Convert that to the dominant Floquet exponent.
        return numpy.log(rho0) / self.period

    def get_stable_age_density(self):
        monodromy = self.solve_monodromy()
        (_, v0) = _utility.get_dominant_eigen(monodromy, which='LM',
                                              return_eigenvector=True)
        # Normalize `v0` in place so that its integral over ages is 1.
        v0 /= scipy.integrate.trapz(v0, self.ages)
        return pandas.Series(v0,
                             index=pandas.Index(self.ages, name='age (y)'),
                             name='stable age distribution')


def get_birth_scaling_for_zero_pop_growth(birth_rate, *args, **kwds):
    '''Find the birth scaling that gives zero population growth rate.'''
    solver = _Solver(birth_rate, *args, **kwds)
    # For a lower limit, we know that birth_scaling = 0 gives
    # `solver.get_pop_growth(0) < 0`,
    # so we need to find an upper limit `upper`
    # with `solver.get_pop_growth(upper) > 0`.
    (lower, upper) = (0, 1)
    MULT = 2
    while solver.get_pop_growth(upper) <= 0:
        (lower, upper) = (upper, MULT * upper)
    return scipy.optimize.brentq(solver.get_pop_growth, lower, upper)


def get_stable_age_density(birth_rate, *args, **kwds):
    '''Find the stable age distribution.'''
    solver = _Solver(birth_rate, *args, **kwds)
    return solver.get_stable_age_density()


def get_death_rate_mean(birth_rate, *args, **kwds):
    stable_age_density = get_stable_age_density(birth_rate, *args, **kwds)
    ages = stable_age_density.index
    death_total = scipy.integrate.trapz(death.rate(ages)
                                        * stable_age_density,
                                        ages)
    density_total = scipy.integrate.trapz(stable_age_density,
                                          ages)
    return death_total / density_total
