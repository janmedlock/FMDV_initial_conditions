'''Solver.'''

import functools

import numpy

from .. import _solver
from .. import _utility
from .._utility import sparse


class Solver(_solver.Base):
    '''Crank–Nicolson solver.'''

    _sparse = True

    def __init__(self, model):
        self.z_step = self.t_step = model.z_step
        super().__init__(model)

    @functools.cached_property
    def Zeros(self):
        '''These are zero matrices of different sizes used in
        constructing the other matrices. `Zeros` is built on first use
        and then reused.'''
        K = len(self.model.z)
        Zeros = {
            '11': sparse.array((1, 1)),
            '1K': sparse.array((1, K)),
            'K1': sparse.array((K, 1)),
            'KK': sparse.array((K, K))
        }
        return Zeros

    def _mXX(self):
        '''Build hXX, tXX, and bXX.'''
        mXX = numpy.array([[1]])
        return mXX

    def _myX(self):
        '''Build tyX and byX.'''
        K = len(self.model.z)
        shape = (K, 1)
        # The first entry is `1 / self.z_step`.
        data = {
            (0, 0): 1 / self.z_step
        }
        myX = sparse.array_from_dict(data, shape=shape)
        return myX

    def _I(self):
        '''Build the identity matrix.'''
        n = len(self.model.states)
        m = len(self.model.states_with_z)
        K = len(self.model.z)
        size = m * K + (n - m)
        I = sparse.identity(size)
        return I

    def _beta(self):
        '''Build the transmission rate vector beta.'''
        K = len(self.model.z)
        ones1K = numpy.ones((1, K))
        zeros = self.Zeros
        beta = (
            self.model.transmission.rate
            * self.z_step
            * sparse.hstack(
                [zeros['1K'], zeros['11'], zeros['1K'], ones1K, zeros['11']]
            )
        )
        return beta

    def _hXX(self):
        '''Build a diagonal block for an X state of H(q).'''
        hXX = self._mXX()
        return hXX

    def _Hyy(self, q):
        '''Build a diagonal block for a y state of H(q).'''
        K = len(self.model.z)
        if q == 'new':
            diags = {
                0: numpy.ones(K)
            }
        elif q == 'cur':
            diags = {
                -1: numpy.ones(K - 1),
                0: numpy.hstack([numpy.zeros(K - 1), 1])
            }
        else:
            raise ValueError(f'{q=}!')
        Hyy = sparse.diags_from_dict(diags)
        return Hyy

    def _H(self, q):
        '''Build the time step matrix H(q).'''
        hXX = self._hXX()
        Hyy = self._Hyy(q)
        H = sparse.block_diag(
            [Hyy, hXX, Hyy, Hyy, hXX]
        )
        return H

    def _fXX(self, pi):
        '''Build a diagonal block for an X state of F(q).'''
        fXX = numpy.array([[pi]])
        return fXX

    def _fXy(self, psi):
        '''Build a block to an X state from a y state of F(q).'''
        K = len(self.model.z)
        if numpy.isscalar(psi):
            psi = psi * numpy.ones(K)
        # `[None]` adds a dimension, going from shape (K, ) to shape (1, K).
        fXy = (self.z_step * psi)[None]
        return fXy

    def _Fyy(self, q, psi):
        '''Build a diagonal block for a y state of F(q).'''
        K = len(self.model.z)
        if numpy.isscalar(psi):
            psi = psi * numpy.ones(K)
        if q == 'new':
            diags = {
                0: psi
            }
        elif q == 'cur':
            diags = {
                -1: psi[:-1],
                0: numpy.hstack([numpy.zeros(K - 1), psi[-1]])
            }
        else:
            raise ValueError(f'{q=}!')
        Fyy = sparse.diags_from_dict(diags)
        return Fyy

    def _Fyz(self, psi):
        '''Build an off-diagonal block between y states of F(q).'''
        K = len(self.model.z)
        shape = (K, K)
        # The first row is `psi`.
        data = {
            (0, (None, )): psi
        }
        Fyz = sparse.array_from_dict(data, shape=shape)
        return Fyz

    def _get_rate(self, which):
        '''Get the rate `which` and make finite any infinite entries.'''
        param = getattr(self.model, which)
        rate = param.rate(self.model.z)
        return _utility.rate_make_finite(rate)

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        mu = self.model.death_rate_mean
        omega = self._get_rate('waning')
        rho = self._get_rate('progression')
        gamma = self._get_rate('recovery')
        fXX = self._fXX
        fXy = self._fXy
        Fyy = functools.partial(self._Fyy, q)
        Fyz = self._Fyz
        F = sparse.bmat([
            [Fyy(- omega - mu), None, None, None, None],
            [fXy(omega), fXX(- mu), None, None, None],
            [None, None, Fyy(- rho - mu), None, None],
            [None, None, Fyz(rho), Fyy(- gamma - mu), None],
            [None, None, None, fXy(gamma), fXX(- mu)]
        ])
        return F

    def _tXX(self):
        '''Build a diagonal block for an X state of T(q).'''
        tXX = self._mXX()
        return tXX

    def _tyX(self):
        '''Build a block to a y state from an X state of T(q).'''
        tyX = self._myX()
        return tyX

    @functools.cached_property
    def _T_(self):
        '''T is independent of q. `_T_` is built on first use and then
        reused.'''
        tXX = self._tXX()
        tyX = self._tyX()
        Zeros = self.Zeros
        T = sparse.bmat([
            [Zeros['KK'], Zeros['K1'], Zeros['KK'], Zeros['KK'], Zeros['K1']],
            [Zeros['1K'],       - tXX, Zeros['1K'], Zeros['1K'], Zeros['11']],
            [Zeros['KK'],         tyX, Zeros['KK'], Zeros['KK'], Zeros['K1']],
            [Zeros['KK'], Zeros['K1'], Zeros['KK'], Zeros['KK'], Zeros['K1']],
            [Zeros['1K'], Zeros['11'], Zeros['1K'], Zeros['1K'], Zeros['11']]
        ])
        return T

    def _T(self, q):
        '''Build the transmission matrix T(q).'''
        T = self._T_
        return T

    def _bXX(self):
        '''Build a diagonal block for an X state of B.'''
        bXX = self._mXX()
        return bXX

    def _bXy(self):
        '''Build a block to an X state from a y state of B.'''
        K = len(self.model.z)
        shape = (1, K)
        bXy = self.z_step * numpy.ones(shape)
        return bXy

    def _byX(self):
        '''Build a block to a y state from an X state of B.'''
        byX = self._myX()
        return byX

    def _B(self):
        '''Build the birth matrix B.'''
        bXX = self._bXX()
        byX = self._byX()
        bXy = self._bXy()
        Zeros = self.Zeros
        B = sparse.bmat([
            [Zeros['KK'], Zeros['K1'], Zeros['KK'], Zeros['KK'],         byX],
            [        bXy,         bXX,         bXy,         bXy, Zeros['11']],
            [Zeros['KK'], Zeros['K1'], Zeros['KK'], Zeros['KK'], Zeros['K1']],
            [Zeros['KK'], Zeros['K1'], Zeros['KK'], Zeros['KK'], Zeros['K1']],
            [Zeros['1K'], Zeros['11'], Zeros['1K'], Zeros['1K'], Zeros['11']]
        ])
        return B
