'''Solver.'''

import functools

from ._population.solver import Mixin
from .. import _model, _utility


class Solver(Mixin, _model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _jacobian_method_default = 'dense'

    def __init__(self, model, **kwds):
        super().__init__(model, model.parameters.age_max, **kwds)

    @functools.cached_property
    def beta(self):
        '''The transmission rate vector.'''
        blocks = [self._zeros_a] * len(self.model.states)
        infectious = self.model.states.index('infectious')
        blocks[infectious] = self._iota_a
        return (
            self.model.parameters.transmission.rate
            * _utility.sparse.hstack(blocks)
        )

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''Build the identity matrix.'''
        return _utility.sparse.block_diag(
            [self._I_a] * len(self.model.states)
        )

    def H(self, q):  # pylint: disable=invalid-name
        '''The time-step matrix, H(q).'''
        return _utility.sparse.block_diag(
            [self._H_a(q)] * len(self.model.states)
        )

    def F(self, q):  # pylint: disable=invalid-name
        '''The transition matrix, F(q).'''
        mu = self.model.parameters.death.rate(self.a)
        omega = 1 / self.model.parameters.waning.mean
        rho = 1 / self.model.parameters.progression.mean
        gamma = 1 / self.model.parameters.recovery.mean
        F_a = functools.partial(self._F_a, q)  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [F_a(- omega - mu), None, None, None, None],
            [F_a(omega), F_a(- mu), None, None, None],
            [None, None, F_a(- rho - mu), None, None],
            [None, None, F_a(rho), F_a(- gamma - mu), None],
            [None, None, None, F_a(gamma), F_a(- mu)]
        ])

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        B_a = self._B_a  # pylint: disable=invalid-name
        Zeros_a_a = self._Zeros_a_a  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [None,      None, None, None, B_a],
            [B_a,       B_a,  B_a,  B_a, None],
            [Zeros_a_a, None, None, None, None],
            [Zeros_a_a, None, None, None, None],
            [Zeros_a_a, None, None, None, None]
        ])

    def _T(self, q):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        H_a = self._H_a(q)  # pylint: disable=invalid-name
        Zeros_a_a = self._Zeros_a_a  # pylint: disable=invalid-name
        return _utility.sparse.bmat([
            [Zeros_a_a, None,  Zeros_a_a, None,      None],
            [None,      - H_a, None,      None,      None],
            [None,      H_a,   None,      None,      None],
            [None,      None,  None,      Zeros_a_a, None],
            [None,      None,  None,      None,      Zeros_a_a]
        ])
