'''Models of population size.'''

import functools

import numpy
import scipy.optimize
import scipy.sparse

from . import _utility


# Common sparse array format.
_SPARSE_ARRAY = scipy.sparse.csr_array


class _Solver:
    '''Solver for the monodromy matrix of a linear age-structured
    model for the population size with age-dependent death rate,
    age-dependent maternity, and periodic time-dependent birth rate.'''

    def __init__(self, birth, death, a_step=0.1, a_max=50):
        self.birth = birth
        self.death = death
        self.a_step = self.t_step = a_step
        self.period = self.birth.period
        if self.period == 0:
            self.period = self.t_step
        self.a = _utility.build_t(0, a_max, self.a_step)
        self.t = _utility.build_t(0, self.period, self.t_step)
        assert numpy.isclose(self.t[-1], self.period)
        self._build_matrices()
        self._check_matrices()

    def _FqXW(self, q, pi):
        J = len(self.a)
        if numpy.isscalar(pi):
            pi = pi * numpy.ones(J)
        if q == 'new':
            diags = {0: pi}
        elif q == 'cur':
            diags = {-1: pi[:-1],
                     0: numpy.hstack([numpy.zeros(J - 1), pi[-1]])}
        else:
            raise ValueError(f'{q=}!')
        FqXW = _utility.sparse.diags(diags)
        return _SPARSE_ARRAY(FqXW)

    def _Hq(self, q):
        return self._FqXW(q, 1)

    def _Fq(self, q):
        mu = self.death.rate(self.a)
        return self._FqXW(q, - mu)

    def _B(self):
        J = len(self.a)
        nu = self.birth.maternity(self.a)
        B = scipy.sparse.lil_array((J, J))
        B[0] = nu
        return _SPARSE_ARRAY(B)

    def _build_matrices(self):
        '''Build the matrices used for stepping forward in time.'''
        self.H_new = self._Hq('new')
        self.H_cur = self._Hq('cur')
        self.F_new = self._Fq('new')
        self.F_cur = self._Fq('cur')
        self.B = self._B()

    def _check_matrices(self):
        assert _utility.is_Z_matrix(self.H_new)
        assert _utility.is_nonnegative(self.H_cur)
        assert _utility.is_Metzler_matrix(self.F_new)
        assert _utility.is_Metzler_matrix(self.B)
        assert _utility.is_nonnegative(self.B)
        HFB_new = (self.H_new
                   - self.t_step / 2 * (self.F_new
                                        + self.birth.rate_max * self.B))
        assert _utility.is_M_matrix(HFB_new)
        HFB_cur = (self.H_cur
                   + self.t_step / 2 * (self.F_cur
                                        + self.birth.rate_min * self.B))
        assert _utility.is_nonnegative(HFB_cur)

    def step(self, t_cur, Phi_cur, birth_scaling, display=False):
        '''Do a step of the solver.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + self.t_step / 2
        b_mid = birth_scaling * self.birth.rate(t_mid)
        HFB_new = (self.H_new
                   - self.t_step / 2 * (self.F_new
                                        + b_mid * self.B))
        HFB_cur = (self.H_cur
                   + self.t_step / 2 * (self.F_cur
                                        + b_mid * self.B))
        return scipy.sparse.linalg.spsolve(HFB_new, HFB_cur @ Phi_cur)

    def monodromy(self, birth_scaling=1, display=False):
        '''Get the monodromy matrix Psi = Phi(T), where Phi is the
        fundmental solution and T is the period.'''
        if len(self.t) == 0:
            return None
        # The initial condition is the identity matrix.
        Phi_new = numpy.identity(len(self.a))
        Phi_cur = numpy.empty_like(Phi_new)
        for t_cur in self.t[:-1]:
            # Update so that what was the new value of the solution is
            # now the current value and what was the current value of
            # the solution will be storage space for the new value.
            (Phi_cur, Phi_new) = (Phi_new, Phi_cur)
            Phi_new[:] = self.step(t_cur, Phi_cur, birth_scaling,
                                   display=display)
        return Phi_new

    def population_growth_rate(self, birth_scaling, display=False):
        '''Get the population growth rate.'''
        Psi = self.monodromy(birth_scaling, display=display)
        # Get the dominant Floquet multiplier.
        rho_dom = _utility.get_dominant_eigen(Psi, which='LM',
                                              return_eigenvector=False)
        # Convert the dominant Floquet multiplier to
        # the dominant Floquet exponent.
        mu_dom = numpy.log(rho_dom) / self.period
        return mu_dom

    def stable_age_density(self, display=False):
        Psi = self.monodromy(display=display)
        (_, v_dom) = _utility.get_dominant_eigen(Psi, which='LM',
                                                 return_eigenvector=True)
        # Normalize `v_dom` in place so that its integral over a is 1.
        v_dom /= v_dom.sum() * self.a_step
        return (self.a, v_dom)


def birth_scaling_for_zero_population_growth(birth, death,
                                             *args, display=False, **kwds):
    '''Find the birth scaling that gives zero population growth rate.'''
    solver = _Solver(birth, death, *args, **kwds)
    # Set the parameter `display`.
    fcn = functools.partial(solver.population_growth_rate,
                            display=display)
    # `_Solver.population_growth_rate()` is increasing in
    # `birth_scaling`. Find a starting bracket `(lower, upper)` with
    # `solver.population_growth_rate(upper) > 0` and
    # `solver.population_growth_rate(lower) < 0`.
    SCALE = 2
    upper = 1  # Starting guess.
    while fcn(upper) < 0:
        upper *= SCALE
    lower = upper / SCALE  # Starting guess.
    while fcn(lower) > 0:
        (lower, upper) = (lower / SCALE, lower)
    scaling = scipy.optimize.brentq(fcn, lower, upper)
    return scaling


def stable_age_density(birth, death,
                       *args, display=False, **kwds):
    '''Find the stable age distribution.'''
    solver = _Solver(birth, death, *args, **kwds)
    return solver.stable_age_density(display=display)
