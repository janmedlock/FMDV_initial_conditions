'''Solver.'''

import functools

import numpy
import scipy.optimize

from . import _integral
from ... import _utility


class Solver:
    '''Crank–Nicolson solver.'''

    def __init__(self, model):
        self.birth = model.birth
        self.death = model.death
        self.t_step = model.t_step
        self.a_step = model.a_step
        self.a_max = model.a_max
        self.period = self.birth.period
        if self.period == 0:
            self.period = self.t_step
        assert self.period > 0
        self._monodromy_initialized = False

    @staticmethod
    def _get_a_step(t_step):
        '''Get the age step.'''
        a_step = t_step
        return a_step

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

    def multiplier_from_growth_rate(self, growth_rate):
        '''Convert a growth rate into a multiplier.'''
        multiplier = numpy.exp(growth_rate * self.period)
        return multiplier

    def growth_rate_from_multiplier(self, multiplier):
        '''Convert a growth rate into a multiplier.'''
        growth_rate = numpy.log(multiplier) / self.period
        return growth_rate

    # TODO: Consider caching this method.
    def population_growth_rate(self, birth_scaling, _guess=None, **kwds):
        '''Get the population growth rate.'''
        Psi = self.monodromy(birth_scaling, **kwds)
        # Get the dominant Floquet multiplier.
        sigma = (self.multiplier_from_growth_rate(_guess)
                 if _guess is not None
                 else None)
        rho_dom = _utility.linalg.get_dominant_eigen(Psi, which='LM',
                                                     sigma=sigma,
                                                     return_eigenvector=False)
        # Convert the dominant Floquet multiplier to
        # the dominant Floquet exponent.
        mu_dom = self.growth_rate_from_multiplier(rho_dom)
        print(f'{birth_scaling=}: {mu_dom=}')
        return mu_dom

    # TODO: Cache this method.
    def birth_scaling_for_zero_population_growth(self, **kwds):
        '''Find the birth scaling that gives zero population growth rate.'''
        # Set keyword arguments. In particular, we are looking for
        # growth rate of 0.
        growth_rate = functools.partial(self.population_growth_rate,
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

    # TODO: Cache this method.
    def stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        Psi = self.monodromy(**kwds)
        # The growth rate is 0.
        sigma = self.multiplier_from_growth_rate(0)
        (rho_dom, v_dom) = _utility.linalg.get_dominant_eigen(
            Psi, which='LM', sigma=sigma, return_eigenvector=True
        )
        assert numpy.isclose(self.growth_rate_from_multiplier(rho_dom), 0)
        # Normalize `v_dom` in place so that its integral over a is 1.
        v_dom /= _integral.over_a(v_dom, self.a_step)
        return (self.a, v_dom)
