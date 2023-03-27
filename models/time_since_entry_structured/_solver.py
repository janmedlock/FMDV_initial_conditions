'''Solver.'''

import functools

import numpy

from .. import _model, _utility


class Solver(_model.solver.Solver):
    '''Crank–Nicolson solver.'''

    _sparse = True

    _jacobian_method_default = 'sparse_csc'

    def __init__(self, model, **kwds):
        self.z = model.z
        super().__init__(model, **kwds)

    @staticmethod
    def _get_z_step(t_step):
        z_step = t_step
        assert z_step > 0
        return z_step

    @property
    def z_step(self):
        return self._get_z_step(self.t_step)

    @functools.cached_property
    def Zeros(self):
        '''These are zero matrices of different sizes used in
        constructing the other matrices. `Zeros` is built on first use
        and then reused.'''
        K = len(self.z)
        Zeros = {
            '11': _utility.sparse.array((1, 1)),
            '1K': _utility.sparse.array((1, K)),
            'K1': _utility.sparse.array((K, 1)),
            'KK': _utility.sparse.array((K, K))
        }
        return Zeros

    @functools.cached_property
    def Iyy(self):
        '''Build an identity matrix block for a y state. `Iyy` is
        built on first use and then reused.'''
        K = len(self.z)
        Iyy = _utility.sparse.identity(K)
        return Iyy

    @functools.cached_property
    def Lyy(self):
        '''Build the lag matrix for a y state. `Lyy` is built on first
        use and then reused.'''
        K = len(self.z)
        diags = {
            -1: numpy.ones(K - 1),
            0: numpy.hstack([numpy.zeros(K - 1), 1])
        }
        Lyy = _utility.sparse.diags_from_dict(diags)
        return Lyy

    @functools.cached_property
    def zeta(self):
        '''Build the vector for entering a y state. `zeta` is
        built on first use and then reused.'''
        K = len(self.z)
        zeta = _utility.sparse.array_from_dict(
            {(0, 0): 1 / self.z_step},
            shape=(K, 1)
        )
        return zeta

    def _sigma(self, xi):
        '''Build the vector to integrate a y state over `xi`.'''
        if numpy.isscalar(xi):
            K = len(self.z)
            xi *= numpy.ones(K)
        sigma = _utility.sparse.array(self.z_step * xi)
        return sigma

    def _mXX(self):
        '''Build hXX, tXX, and bXX.'''
        mXX = _utility.sparse.identity(1)
        return mXX

    def _I(self):
        '''Build the identity matrix.'''
        n = len(self.model.states)
        m = len(self.model.states_with_z)
        K = len(self.z)
        size = m * K + (n - m)
        I = _utility.sparse.identity(size)
        return I

    def _beta(self):
        '''Build the transmission rate vector beta.'''
        K = len(self.z)
        ones1K = numpy.ones((1, K))
        zeros = self.Zeros
        beta = (
            self.model.parameters.transmission.rate
            * self.z_step
            * _utility.sparse.hstack(
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
        if q == 'new':
            Hyy = self.Iyy
        elif q == 'cur':
            Hyy = self.Lyy
        else:
            raise ValueError(f'{q=}!')
        return Hyy

    def _H(self, q):
        '''Build the time step matrix H(q).'''
        hXX = self._hXX()
        Hyy = self._Hyy(q)
        H = _utility.sparse.block_diag(
            [Hyy, hXX, Hyy, Hyy, hXX]
        )
        return H

    def _fXX(self, pi):
        '''Build a diagonal block for an X state of F(q).'''
        fXX = _utility.sparse.array([[pi]])
        return fXX

    def _Fyy(self, q, xi):
        '''Build a diagonal block for a y state of F(q).'''
        if q not in {'new', 'cur'}:
            raise ValueError(f'{q=}!')
        if numpy.isscalar(xi):
            K = len(self.z)
            xi *= numpy.ones(K)
        Fyy = _utility.sparse.diags(xi)
        if q == 'cur':
            Fyy = self.Lyy @ Fyy
        return Fyy

    def _fXy(self, xi):
        '''Build a block to an X state from a y state of F(q).'''
        fXy = self._sigma(xi)
        return fXy

    def _Fyz(self, xi):
        '''Build an off-diagonal block between y states of F(q).'''
        Fyz = self.zeta @ self._sigma(xi)
        return Fyz

    def _get_rate(self, which):
        '''Get the rate `which` and make finite any infinite entries.'''
        param = getattr(self.model.parameters, which)
        rate = param.rate(self.z)
        return _utility.numerical.rate_make_finite(rate)

    def _F(self, q):
        '''Build the transition matrix F(q).'''
        mu = self.model.parameters.death_rate_mean
        omega = self._get_rate('waning')
        rho = self._get_rate('progression')
        gamma = self._get_rate('recovery')
        fXX = self._fXX
        fXy = self._fXy
        Fyy = functools.partial(self._Fyy, q)
        Fyz = self._Fyz
        F = _utility.sparse.bmat([
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
        tyX = self.zeta
        return tyX

    @functools.cached_property
    def _T_(self):
        '''T is independent of q. `_T_` is built on first use and then
        reused.'''
        tXX = self._tXX()
        tyX = self._tyX()
        Zeros = self.Zeros
        T = _utility.sparse.bmat([
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
        K = len(self.z)
        tau = _utility.sparse.array(self.z_step * numpy.ones(shape=(1, K)))
        bXy = tau
        return bXy

    def _byX(self):
        '''Build a block to a y state from an X state of B.'''
        byX = self.zeta
        return byX

    def _B(self):
        '''Build the birth matrix B.'''
        bXX = self._bXX()
        byX = self._byX()
        bXy = self._bXy()
        Zeros = self.Zeros
        B = _utility.sparse.bmat([
            [Zeros['KK'], Zeros['K1'], Zeros['KK'], Zeros['KK'],         byX],
            [        bXy,         bXX,         bXy,         bXy, Zeros['11']],
            [Zeros['KK'], Zeros['K1'], Zeros['KK'], Zeros['KK'], Zeros['K1']],
            [Zeros['KK'], Zeros['K1'], Zeros['KK'], Zeros['KK'], Zeros['K1']],
            [Zeros['1K'], Zeros['11'], Zeros['1K'], Zeros['1K'], Zeros['11']]
        ])
        return B
