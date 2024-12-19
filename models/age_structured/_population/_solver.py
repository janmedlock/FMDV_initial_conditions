'''Solver.'''

import functools

import numpy
import scipy.optimize

from .. import _base
from ... import parameters, _utility


class Solver(_base.Solver):
    '''Crank–Nicolson solver.'''

    sparse = True

    # These methods are slow, so cache their results to disk using
    # `_utility.cache`.
    _methods_cached = ('population_growth_rate',
                       'birth_scaling_for_zero_population_growth',
                       'stable_age_density')

    # For caching, the hash of class instances is restricted to only
    # depend on the value of these attributes by restricting the state
    # produced by `.__getstate__()`.
    _state_attrs = ('parameters', 't_step')

    def __init__(self, model):
        super().__init__(model.t_step,
                         parameters.PopulationParameters(model.parameters))
        self.t_step = model.t_step
        self._checked_matrices = False
        self._cache_methods()

    @functools.cached_property
    def period(self):
        '''Get the period over which to solve.'''
        period = self.parameters.period
        if period is None:
            period = self.t_step
        assert period > 0
        return period

    def _cache_methods(self):
        '''Cache the methods in `self._methods_cached`.'''
        # This is called from both `__init__()` and `__setstate__()`.
        for name in self._methods_cached:
            method = getattr(self, name)
            cached = _utility.cache.cache(method)
            setattr(self, name, cached)

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
        self._cache_methods()

    @functools.cached_property
    def t(self):
        '''The solution times.'''
        t = _utility.numerical.build_t(0, self.period, self.t_step)
        assert numpy.isclose(t[-1], self.period)
        return t

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''
        return self._I_a

    @property
    def _L(self):  # pylint: disable=invalid-name
        '''The lag matrix.'''
        return self._L_a

    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''
        return self._H_a(q)

    def F(self, q):  # pylint: disable=invalid-name
        '''The transition matrix, F(q).'''
        if q not in self._q_vals:
            raise ValueError(f'{q=}!')
        mu = self.parameters.death.rate(self.a)
        F = _utility.sparse.diags(- mu)  # pylint: disable=invalid-name
        if q == 'cur':
            F = self._L @ F  # pylint: disable=invalid-name
        return F

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        return self._B_a

    def _check_matrices(self, is_M_matrix=True):
        '''Check the solver matrices.'''
        if self._checked_matrices:
            return
        super()._check_matrices(is_M_matrix=is_M_matrix)
        self._checked_matrices = True

    # pylint: disable-next=invalid-name
    def step(self, t_cur, Phi_cur, birth_scaling, display=False):
        '''Do a step.'''
        if display:
            t_new = t_cur + self.t_step
            print(f'{t_new=}')
        t_mid = t_cur + self.t_step / 2
        b_mid = birth_scaling * self.parameters.birth.rate(t_mid)
        # pylint: disable-next=invalid-name
        AB_new = self._cn_op('new',
                             self.A['new'],
                             b_mid * self.B)
        # pylint: disable-next=invalid-name
        AB_cur = self._cn_op('cur',
                             self.A['cur'],
                             b_mid * self.B)
        return _utility.linalg.solve(AB_new, AB_cur @ Phi_cur)

    def monodromy(self, birth_scaling=1, display=False):
        '''Get the monodromy matrix Psi = Phi(T), where Phi is the
        fundmental solution and T is the period.'''
        self._check_matrices()
        if len(self.t) == 0:
            return None
        # The initial condition is the identity matrix.
        Phi_new = numpy.identity(len(self.a))  # pylint: disable=invalid-name
        Phi_cur = numpy.empty_like(Phi_new)  # pylint: disable=invalid-name
        for t_cur in self.t[:-1]:
            # Update so that what was the new value of the solution is
            # now the current value and what was the current value of
            # the solution will be storage space for the new value.
            # pylint: disable-next=invalid-name
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
        # pylint: disable-next=invalid-name
        Psi = self.monodromy(birth_scaling, **kwds)
        # Get the dominant Floquet multiplier.
        # If `_guess` is not `None`, find the multiplier closest to
        # `rho_guess`, i.e. the exponent closest to `_guess`.
        # Otherwise, find the multiplier with largest magnitude.
        rho_guess = (self.multiplier_from_exponent(_guess)
                     if _guess is not None
                     else None)
        rho_dom = _utility.linalg.eig_dominant(
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
        scale = 2
        upper = 1.  # Starting guess.
        while growth_rate(upper) < 0:
            upper *= scale
        lower = upper / scale  # Starting guess.
        while growth_rate(lower) > 0:
            (lower, upper) = (lower / scale, lower)
        scaling = scipy.optimize.brentq(growth_rate, lower, upper)
        return scaling

    def integral_over_a(self, arr, *args, **kwds):
        '''Integrate `arr` over age. `args` and `kwds` are passed on
        to `.sum()`.'''
        return arr.sum(*args, **kwds) * self.a_step

    def stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        Psi = self.monodromy(**kwds)  # pylint: disable=invalid-name
        # This method assumes it is being called after birth has been
        # scaled so that the population growth rate is 0.
        growth_rate = 0
        growth_mult = self.multiplier_from_exponent(growth_rate)
        # Find the eigenvector for the multiplier closest to
        # `growth_mult`, i.e. the exponent closest to `growth_rate`.
        (rho_dom, v_dom) = _utility.linalg.eig_dominant(
            Psi,
            which='LM', sigma=growth_mult,
            return_eigenvector=True
        )
        assert numpy.isclose(rho_dom, growth_mult)
        # Normalize `v_dom` in place so that its integral over a is 1.
        v_dom /= self.integral_over_a(v_dom)
        return (self.a, v_dom)
