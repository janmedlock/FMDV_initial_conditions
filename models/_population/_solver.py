'''Models of population size.'''

import numpy
import scipy.integrate
import scipy.optimize
import scipy.sparse

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
        self._init_matrices()

    def _init_matrices(self):
        '''Build the matrices used for stepping forward in time.'''
        J = len(self.ages)
        H0 = _utility.sparse.diags({0: numpy.ones(J)})
        H1 = _utility.sparse.diags({
            -1: numpy.ones(J - 1),
            0: numpy.hstack([numpy.zeros(J - 1), 1]),
        })
        mu = self.death.rate(self.ages)
        F0 = _utility.sparse.diags({0: -mu})
        F1 = _utility.sparse.diags({
            -1: -mu[:-1],
            0: numpy.hstack([numpy.zeros(J - 1), -mu[-1]]),
        })
        self._HF0 = scipy.sparse.csr_array(H0 - self.time_step / 2 * F0)
        self._HF1 = scipy.sparse.csr_array(H1 + self.time_step / 2 * F1)
        B = scipy.sparse.lil_array((J, J))
        B[0] = self.birth.maternity(self.ages)
        self._B = scipy.sparse.csr_array(B)

    def _set_initial_condition(self):
        '''Set the initial condition.'''
        # self._sol_new[:] = numpy.eye(len(self.ages))
        # Avoid building a new matrix.
        self._sol_new[:] = 0
        numpy.fill_diagonal(self._sol_new, 1)

    def _step(self, t_cur, birth_scaling):
        '''Do a step of the solver.'''
        # Update so that what was the new value of the solution is now
        # the current value and what was the current value of the
        # solution will be storage space for the new value.
        (self._sol_cur, self._sol_new) = (self._sol_new, self._sol_cur)
        t_mid = t_cur + self.time_step / 2
        B_mid =  (self.age_step / 2
                  * birth_scaling
                  * self.birth.rate(t_mid)
                  * self._B)
        HFB0 = self._HF0 - B_mid
        HFB1 = self._HF1 + B_mid
        self._sol_new[:] = scipy.sparse.linalg.spsolve(HFB0,
                                                       HFB1 @ self._sol_cur)

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
    scaling = scipy.optimize.brentq(solver.population_growth_rate,
                                    lower, upper)
    return scaling


def stable_age_density(birth, death, *args, **kwds):
    '''Find the stable age distribution.'''
    solver = _Solver(birth, death, *args, **kwds)
    return solver.stable_age_density()
