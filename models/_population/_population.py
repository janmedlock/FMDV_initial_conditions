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
        self._sol_cur = numpy.empty((len(self.ages), ) * 2)
        self._sol_new = numpy.empty((len(self.ages), ) * 2)
        self._init_crank_nicolson()
        self._init_birth()

    def _init_crank_nicolson(self):
        '''Build the matrix used for the Crank–Nicolson step.'''
        ages_mid = (self.ages[1:] + self.ages[:-1]) / 2
        mu = self.death.rate(ages_mid)
        P_diag = numpy.hstack([1,
                               1 + mu * self.time_step / 2])
        Q_subdiag_1 = 1 - mu * self.time_step / 2
        # P = scipy.sparse.diags(P_diag, 0)
        # Q = scipy.sparse.diags(Q_subdiag_1, -1)
        # P_inv = scipy.sparse.linalg.inv(P)
        # C = P_inv @ R  # This will not be a `scipy.sparse.spmatrix()`.
        # Exploit the sparsity of P & Q to directly build C.
        C_subdiag_1 = Q_subdiag_1 / P_diag[1:]
        C = scipy.sparse.diags(C_subdiag_1, -1)
        # Keep the last age group from ageing out.
        mu_end = self.death.rate(self.ages[-1])
        P_end = 1 + mu_end * self.time_step / 2
        Q_end = 1 - mu_end * self.time_step / 2
        C = C.tolil()
        C[-1, -1] += Q_end / P_end
        self._CN = _sparse.csr_array(C)

    def _init_birth(self):
        '''Build the vector used for the integral step.'''
        v = self.age_step * numpy.ones(len(self.ages))
        v[[0, -1]] /= 2
        nu = self.birth.maternity(self.ages)
        self._birth_integral = v * nu / 2
        # Temporary storage for efficiency.
        self._temp_cur = numpy.empty(len(self.ages))
        self._temp_new = numpy.empty(len(self.ages))

    def _set_initial_condition(self):
        '''Set the initial condition.'''
        # self._sol_new[:] = numpy.eye(len(self.ages))
        # Avoid building a new matrix.
        self._sol_new[:] = 0
        numpy.fill_diagonal(self._sol_new, 1)

    def _step_crank_nicolson(self):
        '''Do the Crank–Nicolson step.'''
        # self._sol_new[:] = self._CN @ self._sol_cur
        # Avoid building a new matrix.
        self._sol_new[:] = 0
        # self._sol_new += self._CN @ self._sol_cur
        self._CN.matvecs(self._sol_cur, self._sol_new)

    def _step_birth(self, t_cur, birth_scaling):
        '''Do the birth step.'''
        t_mid = t_cur + self.time_step / 2
        # self._sol_new[0] = (birth_scaling
        #                     * self.birth.rate(t_mid)
        #                     * (self._birth_integral @ self._sol_cur
        #                        + self._birth_integral @ self._sol_new))
        # Avoid building new vectors.
        # self._temp_cur[:] = self._birth_integral @ self._sol_cur
        numpy.dot(self._birth_integral, self._sol_cur,
                  out=self._temp_cur)
        # self._temp_new[:] = self._birth_integral @ self._sol_new
        numpy.dot(self._birth_integral, self._sol_new,
                  out=self._temp_new)
        self._temp_new += self._temp_cur
        self._temp_new *= birth_scaling * self.birth.rate(t_mid)
        self._sol_new[0] = self._temp_new

    def _step(self, t_cur, birth_scaling):
        '''Do a step of the solver.'''
        # Update so that what was the new value of the solution is now
        # the current value and what was the current value of the
        # solution will be storage space for the new value.
        (self._sol_cur, self._sol_new) = (self._sol_new, self._sol_cur)
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
        # Convert the dominant Floquet multiplier to
        # the dominant Floquet exponent.
        mu0 = numpy.log(rho0) / self.period
        return mu0

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


def stable_age_density(birth, death, *args, **kwds):
    '''Find the stable age distribution.'''
    solver = _Solver(birth, death, *args, **kwds)
    return solver.stable_age_density()
