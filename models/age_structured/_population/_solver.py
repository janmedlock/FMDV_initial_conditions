'''Solver.'''

import functools

import numpy
import scipy.optimize

from . import _integral
from ... import _utility


class Solver:
    '''Crank–Nicolson solver.'''

    _methods_to_cache = ('_population_growth_rate',
                         '_birth_scaling_for_zero_population_growth',
                         '_stable_age_density')

    def __init__(self, birth, death, t_step, a_max):
        self.birth = birth
        self.death = death
        self.t_step = t_step
        self.a_max = a_max
        self.a_step = self._get_a_step(self.t_step)
        self._monodromy_initialized = False
        self._init_cached()

    def _init_cached(self):
        # Instance attributes that would change the output of the
        # cached methods.
        args_cache = (self.birth, self.death, self.t_step, self.a_max)
        for name in self._methods_to_cache:
            method = getattr(self, name)
            # Cache the method and fix the instance attributes they
            # depend on.
            cached = functools.partial(
                _utility.cache.cache.cache(method, ignore=['self']),
                *args_cache
            )
            setattr(self, name, cached)

    @staticmethod
    def _get_a_step(t_step):
        '''Get the age step.'''
        a_step = t_step
        return a_step

    @property
    def period(self):
        '''The period over which to solve.'''
        period = self.birth.period
        if period == 0:
            period = self.t_step
        assert period > 0
        return period

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
        HXX = _utility.sparse.diags_from_dict(diags)
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
        F = _utility.sparse.diags_from_dict(diags)
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
        B = _utility.sparse.array_from_dict(data, shape=shape)
        return B

    def _build_matrices(self):
        '''Build the matrices used for stepping forward in time.'''
        q_vals = ('new', 'cur')
        self.H = {q: self._H(q) for q in q_vals}
        self.F = {q: self._F(q) for q in q_vals}
        self.B = self._B()

    def _check_matrices(self):
        '''Check the solver matrices.'''
        assert _utility.linalg.is_Z_matrix(self.H['new'])
        assert _utility.linalg.is_nonnegative(self.H['cur'])
        assert _utility.linalg.is_Metzler_matrix(self.F['new'])
        assert _utility.linalg.is_Metzler_matrix(self.B)
        assert _utility.linalg.is_nonnegative(self.B)
        HFB_new = (self.H['new']
                   - self.t_step / 2 * (self.F['new']
                                        + self.birth.rate_max * self.B))
        assert _utility.linalg.is_M_matrix(HFB_new)
        HFB_cur = (self.H['cur']
                   + self.t_step / 2 * (self.F['cur']
                                        + self.birth.rate_min * self.B))
        assert _utility.linalg.is_nonnegative(HFB_cur)

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
        return _utility.linalg.solve(HFB_new, HFB_cur @ Phi_cur)

    def _monodromy_init(self):
        '''Initialize matrices etc needed by `monodromy`.'''
        self.t = _utility.numerical.build_t(0, self.period, self.t_step)
        assert numpy.isclose(self.t[-1], self.period)
        self.a = _utility.numerical.build_t(0, self.a_max, self.a_step)
        self._build_matrices()
        self._check_matrices()
        self._monodromy_initialized = True

    def monodromy(self, birth_scaling=1, display=False):
        '''Get the monodromy matrix Psi = Phi(T), where Phi is the
        fundmental solution and T is the period.'''
        if not self._monodromy_initialized:
            self._monodromy_init()
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

    def multiplier_from_exponent(self, exponent):
        '''Convert a Floquet exponent into a Floquet multiplier.'''
        multiplier = numpy.exp(exponent * self.period)
        return multiplier

    def exponent_from_multiplier(self, multiplier):
        '''Convert a Floquet exponent into a Floquet multiplier.'''
        exponent = numpy.log(multiplier) / self.period
        return exponent

    def _population_growth_rate(self,
                                birth, death, t_step, a_max,
                                birth_scaling, _guess=None, **kwds):
        '''Get the population growth rate, cached version.'''
        Psi = self.monodromy(birth_scaling, **kwds)
        # Get the dominant Floquet multiplier.
        sigma = (self.multiplier_from_exponent(_guess)
                 if _guess is not None
                 else None)
        rho_dom = _utility.linalg.get_dominant_eigen(Psi, which='LM',
                                                     sigma=sigma,
                                                     return_eigenvector=False)
        mu_dom = self.exponent_from_multiplier(rho_dom)
        return mu_dom

    def population_growth_rate(self, birth_scaling, **kwds):
        '''Get the population growth rate.'''
        return self._population_growth_rate(birth_scaling, **kwds)

    def _birth_scaling_for_zero_population_growth(self,
                                                  birth, death, t_step, a_max,
                                                  **kwds):
        '''Find the birth scaling that gives zero population growth
        rate, cached version.'''
        # Set arguments that do not vary. In particular, we are
        # looking for growth rate of 0.
        growth_rate = functools.partial(self._population_growth_rate,
                                        _guess=0, **kwds)
        # `.population_growth_rate()` is increasing in
        # `birth_scaling`. Find a starting bracket `(lower, upper)` with
        # `.population_growth_rate(upper) > 0` and
        # `.population_growth_rate(lower) < 0`.
        SCALE = 2
        upper = 1  # Starting guess.
        while growth_rate(upper) < 0:
            upper *= SCALE
        lower = upper / SCALE  # Starting guess.
        while growth_rate(lower) > 0:
            (lower, upper) = (lower / SCALE, lower)
        scaling = scipy.optimize.brentq(growth_rate, lower, upper)
        return scaling

    def birth_scaling_for_zero_population_growth(self, **kwds):
        '''Find the birth scaling that gives zero population growth rate.'''
        return self._birth_scaling_for_zero_population_growth(**kwds)

    def _stable_age_density(self,
                            birth, death, t_step, a_max,
                            **kwds):
        '''Get the stable age density, cached version.'''
        # This method assumes it is being called after birth has been
        # scaled so that the population growth rate is 0.
        growth_rate = 0
        Psi = self.monodromy(**kwds)
        sigma = self.multiplier_from_exponent(growth_rate)
        (rho_dom, v_dom) = _utility.linalg.get_dominant_eigen(
            Psi, which='LM', sigma=sigma, return_eigenvector=True
        )
        assert numpy.isclose(rho_dom, sigma)
        # Normalize `v_dom` in place so that its integral over a is 1.
        v_dom /= _integral.over_a(v_dom, self.a_step)
        return (self.a, v_dom)

    def stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        return self._stable_age_density(**kwds)
