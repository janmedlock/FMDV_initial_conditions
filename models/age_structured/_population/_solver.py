'''Solver.'''

import functools

import numpy
import scipy.optimize

from ... import _utility


class Solver:
    '''Crank–Nicolson solver.'''

    # These methods are slow, so cache their results to disk using
    # `_utility.cache`.
    _methods_cached = ('population_growth_rate',
                       'birth_scaling_for_zero_population_growth',
                       'stable_age_density')

    # For caching, the hash of class instances is restricted to only
    # depend on the value of these attributes by restricting the state
    # produced by `.__getstate__()`.
    _state_attrs = ('birth', 'death', 't_step', 'a_max')

    def __init__(self, model):
        self.birth = model.parameters.birth
        self.death = model.parameters.death
        self.t_step = model.t_step
        self.a_max = model.a_max
        self._init_post()

    def _init_cached(self):
        '''Cache the methods in `self._methods_cached`.'''
        for name in self._methods_cached:
            method = getattr(self, name)
            cached = _utility.cache.cache(method)
            setattr(self, name, cached)

    def _init_post(self):
        '''Final initialization.'''
        # This is called by both `__init__()` and `__setstate__()`.
        self._monodromy_initialized = False
        self._init_cached()

    def __getstate__(self):
        '''Restrict the state to only the input variables used in
        initializing instances.'''
        state = {name: getattr(self, name)
                 for name in self._state_attrs}
        return state

    def __setstate__(self, state):
        '''Build an instance from the restricted `state` produced by
        `self.__getstate__()`.'''
        self.__dict__.update(state)
        self.__post_init__()

    @staticmethod
    def _get_a_step(t_step):
        '''Get the age step.'''
        a_step = t_step
        return a_step

    @property
    def a_step(self):
        return self._get_a_step(self.t_step)

    @property
    def period(self):
        '''The period over which to solve.'''
        period = self.birth.period
        if period is None:
            period = self.t_step
        assert period > 0
        return period

    def _I(self):
        '''Build the identity matrix.'''
        J = len(self.a)
        I = _utility.sparse.identity(J)
        return I

    def _L(self):
        '''Build the lag matrix.'''
        J = len(self.a)
        diags = {
            -1: numpy.ones(J - 1),
            0: numpy.hstack([numpy.zeros(J - 1), 1])
        }
        L = _utility.sparse.diags_from_dict(diags)
        return L

    def _H(self, q):
        '''Build the time step matrix H(q).'''
        if q == 'new':
            H = self.I
        elif q == 'cur':
            H = self.L
        else:
            raise ValueError(f'{q=}!')
        return H

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        if q not in {'new', 'cur'}:
            raise ValueError(f'{q=}!')
        mu = self.death.rate(self.a)
        F = _utility.sparse.diags(- mu)
        if q == 'cur':
            F = self.L @ F
        return F

    def _B(self):
        '''Build matrices needed by the solver.'''
        J = len(self.a)
        nu = self.birth.maternity(self.a)
        tau = _utility.sparse.array(self.a_step * nu)
        b = _utility.sparse.array_from_dict(
            {(0, 0): 1 / self.a_step},
            shape=(J, 1)
        )
        B = b @ tau
        return B

    def _build_matrices(self):
        '''Build the matrices used for stepping forward in time.'''
        q_vals = ('new', 'cur')
        self.I = self._I()
        self.L = self._L()
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

    def population_growth_rate(self, birth_scaling,
                               _guess=None, **kwds):
        '''Get the population growth rate.'''
        Psi = self.monodromy(birth_scaling, **kwds)
        # Get the dominant Floquet multiplier.
        # If `_guess` is not `None`, find the multiplier closest to
        # `rho_guess`, i.e. the exponent closest to `_guess`.
        # Otherwise, find the multiplier with largest magnitude.
        rho_guess = (self.multiplier_from_exponent(_guess)
                     if _guess is not None
                     else None)
        rho_dom = _utility.linalg.get_dominant_eigen(
            Psi,
            which='LM', sigma=rho_guess,
            return_eigenvector=False
        )
        mu_dom = self.exponent_from_multiplier(rho_dom)
        return mu_dom

    def birth_scaling_for_zero_population_growth(self, **kwds):
        '''Find the birth scaling that gives zero population growth
        rate.'''
        # Set the arguments to `.population_growth_rate()` except for
        # `birth_scaling`, which will be found by optimization. In
        # particular, set the growth rate to be near 0.
        growth_rate = functools.partial(self.population_growth_rate,
                                        _guess=0,
                                        **kwds)
        # `.population_growth_rate()` is increasing in
        # `birth_scaling`. Find a starting bracket `(lower, upper)` with
        # `.population_growth_rate(upper) > 0` and
        # `.population_growth_rate(lower) < 0`.
        SCALE = 2
        upper = 1.  # Starting guess.
        while growth_rate(upper) < 0:
            upper *= SCALE
        lower = upper / SCALE  # Starting guess.
        while growth_rate(lower) > 0:
            (lower, upper) = (lower / SCALE, lower)
        scaling = scipy.optimize.brentq(growth_rate, lower, upper)
        return scaling

    def integral_over_a(self, arr, *args, **kwds):
        '''Integrate `arr` over age. `args` and `kwds` are passed on
        to `.sum()`.'''
        return arr.sum(*args, **kwds) * self.a_step

    def stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        Psi = self.monodromy(**kwds)
        # This method assumes it is being called after birth has been
        # scaled so that the population growth rate is 0.
        growth_rate = 0
        growth_mult = self.multiplier_from_exponent(growth_rate)
        # Find the eigenvector for the multiplier closest to
        # `growth_mult`, i.e. the exponent closest to `growth_rate`.
        (rho_dom, v_dom) = _utility.linalg.get_dominant_eigen(
            Psi, which='LM', sigma=growth_mult, return_eigenvector=True
        )
        assert numpy.isclose(rho_dom, growth_mult)
        # Normalize `v_dom` in place so that its integral over a is 1.
        v_dom /= self.integral_over_a(v_dom)
        return (self.a, v_dom)
