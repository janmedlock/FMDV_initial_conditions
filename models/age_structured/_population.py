'''Models of population size.'''

import functools

import numpy
import scipy.optimize

from .._utility import linalg, numerical, sparse


# Default age step.
_A_STEP = 0.1

# Default age maximum.
_A_MAX = 50


def integral_over_a(arr, a_step=_A_STEP, a_max=_A_MAX,
                    *args, display=None, **kwds):
    '''Integrate `arr` over age.'''
    # The arguments mirror those of _Solver() and
    # stable_age_density(). `a_step` is used, `a_max` and `display`
    # are not used, and `args` and `kwds` are passed on to `.sum()`.
    return arr.sum(*args, **kwds) * a_step


class _Solver:
    '''Solver for the monodromy matrix of a linear age-structured
    model for the population size with age-dependent death rate,
    age-dependent maternity, and periodic time-dependent birth rate.'''

    def __init__(self, birth, death, a_step=_A_STEP, a_max=_A_MAX):
        self.birth = birth
        self.death = death
        self.a_step = self.t_step = a_step
        self.period = self.birth.period
        if self.period == 0:
            self.period = self.t_step
        self.a = numerical.build_t(0, a_max, self.a_step)
        self.t = numerical.build_t(0, self.period, self.t_step)
        assert numpy.isclose(self.t[-1], self.period)
        self._build_matrices()
        self._check_matrices()

    def _H(self, q):
        '''Build the time step matrix H(q).'''
        J = len(self.a)
        if q == 'new':
            diags = {
                0: numpy.ones(J)
            }
        elif q == 'cur':
            diags = {
                -1: numpy.ones(J - 1),
                0: numpy.hstack([numpy.zeros(J - 1), 1])
            }
        else:
            raise ValueError(f'{q=}!')
        HXX = sparse.diags_from_dict(diags)
        return HXX

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        J = len(self.a)
        mu = self.death.rate(self.a)
        if q == 'new':
            diags = {
                0: - mu
            }
        elif q == 'cur':
            diags = {
                -1: - mu[:-1],
                0: numpy.hstack([numpy.zeros(J - 1), - mu[-1]])
            }
        else:
            raise ValueError(f'{q=}!')
        F = sparse.diags_from_dict(diags)
        return F

    def _B(self):
        '''Build matrices needed by the solver.'''
        J = len(self.a)
        shape = (J, J)
        nu = self.birth.maternity(self.a)
        # The first row is `nu`.
        data = {
            (0, (None, )): nu
        }
        B = sparse.array_from_dict(data, shape=shape)
        return B

    def _build_matrices(self):
        '''Build the matrices used for stepping forward in time.'''
        q_vals = ('new', 'cur')
        self.H = {q: self._H(q) for q in q_vals}
        self.F = {q: self._F(q) for q in q_vals}
        self.B = self._B()

    def _check_matrices(self):
        '''Check the solver matrices.'''
        assert linalg.is_Z_matrix(self.H['new'])
        assert linalg.is_nonnegative(self.H['cur'])
        assert linalg.is_Metzler_matrix(self.F['new'])
        assert linalg.is_Metzler_matrix(self.B)
        assert linalg.is_nonnegative(self.B)
        HFB_new = (self.H['new']
                   - self.t_step / 2 * (self.F['new']
                                        + self.birth.rate_max * self.B))
        assert linalg.is_M_matrix(HFB_new)
        HFB_cur = (self.H['cur']
                   + self.t_step / 2 * (self.F['cur']
                                        + self.birth.rate_min * self.B))
        assert linalg.is_nonnegative(HFB_cur)

    def step(self, t_cur, Phi_cur, birth_scaling, display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + self.t_step / 2
        b_mid = birth_scaling * self.birth.rate(t_mid)
        HFB_new = (self.H['new']
                   - self.t_step / 2 * (self.F['new']
                                        + b_mid * self.B))
        HFB_cur = (self.H['cur']
                   + self.t_step / 2 * (self.F['cur']
                                        + b_mid * self.B))
        return linalg.solve(HFB_new, HFB_cur @ Phi_cur)

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
        rho_dom = linalg.get_dominant_eigen(Psi, which='LM',
                                            return_eigenvector=False)
        # Convert the dominant Floquet multiplier to
        # the dominant Floquet exponent.
        mu_dom = numpy.log(rho_dom) / self.period
        return mu_dom

    def stable_age_density(self, display=False):
        '''Get the stable age density.'''
        Psi = self.monodromy(display=display)
        (_, v_dom) = linalg.get_dominant_eigen(Psi, which='LM',
                                               return_eigenvector=True)
        # Normalize `v_dom` in place so that its integral over a is 1.
        v_dom /= integral_over_a(v_dom, a_step=self.a_step)
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
