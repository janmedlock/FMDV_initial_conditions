'''Models of population size.'''

import numpy
import scipy.integrate
import scipy.optimize
import scipy.sparse

from . import _sparse
from .. import _utility


class _Solver:
    '''Solver for the monodromy matrix of a linear age-structured
    model for the population size with age-dependent death rate,
    age-dependent maternity, and periodic time-dependent birth rate.'''

    def __init__(self, birth, death, age_step=0.1, age_max=50):
        self.birth = birth
        self.death = death
        self.age_step = self.time_step = age_step
        self.period = self.birth.period
        if self.period == 0:
            self.period = self.time_step
        self.ages = _utility.build_t(0, age_max, self.age_step)
        self.times = _utility.build_t(0, self.period, self.time_step)
        assert numpy.isclose(self.times[-1], self.period)
        self._sol_new = numpy.empty((len(self.ages), ) * 2)
        self._sol_cur = numpy.empty((len(self.ages), ) * 2)
        self._init_crank_nicolson()
        self._init_birth()

    def _init_crank_nicolson(self):
        '''Build the matrix used for the Crank–Nicolson step.'''
        mat_cn = scipy.sparse.lil_array((len(self.ages), ) * 2)
        ages_mid = (self.ages[1:] + self.ages[:-1]) / 2
        mu = self.death.rate(ages_mid)
        A_diag = 1 + mu * self.time_step / 2
        B_subdiag = 1 - mu * self.time_step / 2
        # Set the first subdiagonal.
        mat_cn.setdiag(B_subdiag / A_diag, -1)
        # Keep the last age group from ageing out.
        mu_end = self.death.rate(self.ages[-1])
        mat_cn[-1, -1] = ((1 - mu_end * self.time_step / 2)
                          / (1 + mu_end * self.time_step / 2))
        self._mat_cn = _sparse.csr_array(mat_cn)

    def _init_birth(self):
        '''Build the vector used for the integral step.'''
        v = self.age_step * numpy.ones(len(self.ages))
        v[[0, -1]] /= 2
        nu = self.birth.maternity(self.ages)
        self._vec_birth = v * nu
        # Temporary storage for efficiency.
        self._vec_temp = numpy.empty(len(self.ages))

    def _set_initial_condition(self):
        '''Set the initial condition.'''
        # self._sol_new[:] = numpy.eye(len(self.ages))
        # Avoid building a new matrix.
        self._sol_new[:] = 0
        numpy.fill_diagonal(self._sol_new, 1)

    def _step_crank_nicolson(self):
        '''Do the Crank–Nicolson step.'''
        # self._sol_new[:] = self._mat_cn @ self._sol_cur
        # Avoid building a new matrix.
        self._sol_new[:] = 0
        # self._sol_new += self._mat_cn @ self._sol_cur
        self._mat_cn.matvecs(self._sol_cur, self._sol_new)

    def _step_birth(self, t_cur, birth_scaling):
        '''Do the birth step.'''
        t_mid = t_cur + self.time_step / 2
        # self._sol_new[0] = (birth_scaling
        #                     * self.birth.rate(t_mid)
        #                     * self._vec_birth @ self._sol_new)
        # Avoid building new vectors.
        self._vec_temp[:] = self._vec_birth
        self._vec_temp *= birth_scaling * self.birth.rate(t_mid) / 2
        self._vec_temp.dot(self._sol_new + self._sol_cur,
                           out=self._sol_new[0])

    def _step(self, t_cur, birth_scaling):
        '''Do a step of the solver.'''
        # Update so that what was the new value of the solution is now
        # the current value and what was the current value of the
        # solution will be storage space for the new value.
        (self._sol_new, self._sol_cur) = (self._sol_cur, self._sol_new)
        self._step_crank_nicolson()
        self._step_birth(t_cur, birth_scaling)

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
        # Convert that to the dominant Floquet exponent.
        return numpy.log(rho0) / self.period

    def stable_age_density(self):
        monodromy = self.solve_monodromy()
        (_, v0) = _utility.get_dominant_eigen(monodromy, which='LM',
                                              return_eigenvector=True)
        # Normalize `v0` in place so that its integral over ages is 1.
        v0 /= scipy.integrate.trapz(v0, self.ages)
        return (self.ages, v0)


def birth_scaling_for_zero_population_growth(birth, death, *args, **kwds):
    '''Find the birth scaling that gives zero population growth rate.'''
    solver = _Solver(birth, death, *args, **kwds)
    # For a lower limit, we know that birth_scaling = 0 gives
    # `solver.population_growth_rate(0) < 0`,
    # so we need to find an upper limit `upper`
    # with `solver.population_growth_rate(upper) > 0`.
    (lower, upper) = (0, 1)
    MULT = 2
    while solver.population_growth_rate(upper) <= 0:
        (lower, upper) = (upper, MULT * upper)
    return scipy.optimize.brentq(solver.population_growth_rate, lower, upper)


def stable_age_density(birth, death, *args, **kwds):
    '''Find the stable age distribution.'''
    solver = _Solver(birth, death, *args, **kwds)
    return solver.stable_age_density()
