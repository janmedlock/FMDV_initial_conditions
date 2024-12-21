'''Jacobian calculators.'''

import functools

import numpy
import scipy.sparse

from . import _crank_nicolson
from .. import age_structured
from .. import _utility


class _MemoizeByID:  # pylint: disable=too-few-public-methods
    '''Decorator to memoize a function `func(obj)` using the key
    `id(obj)`.'''

    def __init__(self, func):
        self.func = func
        self.cache = {}
        functools.update_wrapper(self, func)

    def __call__(self, obj):
        key = id(obj)
        if key not in self.cache:
            self.cache[key] = self.func(obj)
        return self.cache[key]


class Base(_crank_nicolson.Mixin):
    '''Calculate the Jacobian using the matrices from `Solver()`.'''

    name = 'base'

    def __init__(self, solver):
        self._solver = solver
        # Memoize so that identical matrices are only converted once.
        self._convert_arr = _MemoizeByID(self._convert_arr_)

    @property
    def model(self):
        '''The underlying model.'''
        return self._solver.model

    @property
    def t_step(self):
        '''The time step.'''
        return self._solver.t_step

    def _convert_arr_(self, arr, out=None):
        '''No-op that subclasses can override to convert `arr` to the
        desired format. This is wrapped by `_MemoizeByID()` to make
        `._convert_arr()`.'''
        if out is not None:
            out[:] = arr[:]
        return arr

    def _convert_obj(self, obj):
        '''Convert `obj` to the desired format. `obj` can be a
        `numpy.ndarray()` or a `scipy.sparse.sparray()`, or a
        `dict()` whose values are either of those.'''
        if isinstance(obj, (numpy.ndarray, scipy.sparse.sparray)):
            val = self._convert_arr(obj)
        elif isinstance(obj, dict):
            # Recursively convert dict values.
            val = {key: self._convert_obj(val)
                   for (key, val) in obj.items()}
        else:
            raise TypeError
        return val

    @functools.cached_property
    def beta(self):
        '''The transmission rate.'''
        return self._convert_obj(self._solver.beta)

    @functools.cached_property
    def I(self):  # pylint: disable=invalid-name  # noqa: E743
        '''The identity matrix.'''
        return self._convert_obj(self._solver.I)

    @functools.cached_property
    def A(self):  # pylint: disable=invalid-name
        '''The matrices A(q).'''
        return self._convert_obj(self._solver.A)

    @functools.cached_property
    def B(self):  # pylint: disable=invalid-name
        '''The birth matrix, B.'''
        return self._convert_obj(self._solver.B)

    @functools.cached_property
    def T(self):  # pylint: disable=invalid-name
        '''The transmission matrix, T(q).'''
        return self._convert_obj(self._solver.T)

    def _make_column_vector(self, y):
        '''Convert `y` with shape (n, ) to shape (n, 1).'''
        assert numpy.ndim(y) == 1
        return numpy.asarray(y)[:, None]

    def _M(self, q, y_q, b_mid, out=None):  # pylint: disable=invalid-name
        '''Calculate the matrix
        M_q = A_q ± t_step / 2 (b_mid * B
                                + beta @ y_q @ T_q
                                + T_q @ y_q @ beta).'''
        A_q = self.A[q]  # pylint: disable=invalid-name
        # The linear algebra is easier if `y_q` has shape (n, 1)
        # instead of just (n, ).
        y_q = self._make_column_vector(y_q)
        # For models with age structure, `T @ y @ beta` is *much* less
        # sparse.
        # pylint: disable-next=invalid-name
        BT_q = (
            b_mid * self.B
            + self.beta @ y_q * self.T[q]
            + self.T[q] @ y_q @ self.beta
        )
        # M_q = A_q ± self.t_step / 2 * BT_q
        return self._cn_op(q, A_q, BT_q,
                           out=out)

    def _D(self, t_cur, y_cur, y_new):  # pylint: disable=invalid-name
        '''Calculate the D matrix at `t_cur`, given `y_cur` and `y_new`.
        D is the derivative of `y_new` with respect to `y_cur`,
        i.e. D = I + t_step * J.'''
        # Compute `D`, the derivative of `y_cur` with respect to `y_new`,
        # which is `M_new @ D = M_cur`.
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.parameters.birth.rate(t_mid)
        M_cur = self._M('cur', y_cur, b_mid)  # pylint: disable=invalid-name
        M_new = self._M('new', y_new, b_mid)  # pylint: disable=invalid-name
        return _utility.linalg.solve(M_new, M_cur,
                                     overwrite_a=True,
                                     overwrite_b=True)

    def _J(self, D):  # pylint: disable=invalid-name
        '''Calculate the Jacobian
        J = 1 / t_step * (D - I).'''
        return (D - self.I) / self.t_step

    def calculate(self, t_cur, y_cur, y_new):
        '''Calculate the Jacobian at `t_cur`, given `y_cur` and `y_new`.'''
        # This calculation is split so that `._J()` can be
        # implemented in multiple ways while reusing `._D()`.
        D = self._D(t_cur, y_cur, y_new)  # pylint: disable=invalid-name
        return self._J(D)


class Dense(Base):
    '''Jacobian caclulator using dense `numpy.ndarray` matrices.'''

    name = 'dense'

    @staticmethod
    def _empty(shape, dtype=float):
        '''Get an empty array.'''
        arr = numpy.empty(shape, dtype=dtype)
        return arr

    def _convert_arr_(self, arr, out=None):
        '''Convert `arr` to a dense `numpy.ndarray` array.'''
        if isinstance(arr, numpy.ndarray):
            if out is not None:
                out[:] = arr[:]
            val = arr
        elif isinstance(arr, scipy.sparse.sparray):
            val = arr.toarray(out=out)
        else:
            raise TypeError
        return val

    def _init_temp(self):
        '''Initialize temporary storage.'''
        shape = self.I.shape
        self.temp = self._empty(shape)
        # pylint: disable-next=invalid-name
        self.M = {q: self._empty(shape)
                  for q in self._q_vals}

    def __init__(self, solver):
        super().__init__(solver)
        self._init_temp()

    def _M(self, q, y_q, b_mid, out=None):
        '''Calculate the matrix
        M_q = A_q ± t_step / 2 (b_mid * B
                                + beta @ y_q @ T_q
                                + T_q @ y_q @ beta).'''
        if out is None:
            out = self.M[q]
        # Use temporary storage to calculate
        # BT_q = self.t_step / 2 * (b_mid * self.B
        #                           + self.beta @ y_q * self.T[q]
        #                           + self.T[q] @ y_q @ self.beta)
        #
        # Tybeta_q = (T_q @ y_q) @ beta
        # pylint: disable-next=invalid-name
        Ty_q = numpy.dot(self.T[q], y_q,
                         out=out[0])
        # For models with age structure, `Ty_q @ beta` is *much* less
        # sparse.
        # pylint: disable-next=invalid-name
        Tybeta_q = numpy.outer(Ty_q, self.beta,
                               out=self.temp)
        # out[0] = Ty_q is free now.
        BT_q = Tybeta_q  # pylint: disable=invalid-name
        # self.temp = BT_q is *not* free now.
        # betayT_q = (beta @ y_q) * T_q
        betay_q = numpy.inner(self.beta[0], y_q)
        # pylint: disable-next=invalid-name
        betayT_q = numpy.multiply(betay_q, self.T[q],
                                  out=out)
        BT_q += betayT_q  # pylint: disable=invalid-name
        # out = betayT_q is free now.
        # bB_mid = b_mid * B
        # pylint: disable-next=invalid-name
        bB_mid = numpy.multiply(b_mid, self.B,
                                out=out)
        BT_q += bB_mid  # pylint: disable=invalid-name
        # out = bB_mid is free now.
        A_q = self.A[q]  # pylint: disable=invalid-name
        # M_q = A_q ± self.t_step / 2 * BT_q
        # temp is free after this.
        return self._cn_op(q, A_q, BT_q,
                           out=out)

    def _J(self, D):  # pylint: disable=invalid-name
        '''Calculate the Jacobian
        J = 1 / t_step * (D - I).'''
        # Overwrite `D` to store `J`.
        J = D  # pylint: disable=invalid-name
        # J = D - I
        J -= self.I  # pylint: disable=invalid-name
        # J = (D - I) / t_step
        J /= self.t_step  # pylint: disable=invalid-name
        return J


class DenseMemmap(Dense):
    '''Jacobian caclulator using memmapped dense `numpy.ndarray`
    matrices.'''

    name = 'dense_memmap'

    @staticmethod
    def _empty(shape, dtype=float):
        '''Get an empty array.'''
        memmap = _utility.numerical.memmaptemp(shape=shape,
                                               dtype=dtype)
        return memmap

    def _convert_arr_(self, arr, out=None):
        '''Convert `arr` to a memmapped dense `numpy.ndarray` matrix.'''
        if out is None:
            out = self._empty(numpy.shape(arr), dtype=numpy.dtype(arr))
        return super()._convert_arr_(arr, out=out)


class Sparse(Base):
    '''Jacobian caclulator using the default sparse matrices.'''

    name = 'sparse'

    _Array = _utility.sparse.Array

    def _convert_arr_(self, arr, out=None):
        '''Convert `arr` to the desired sparse format.'''
        if out is not None:
            raise ValueError(f'{out=}')
        return self._Array(arr)

    def _make_column_vector(self, y):
        '''Convert `y` with shape (n, ) to shape (n, 1).'''
        return self._Array(super()._make_column_vector(y))


class SparseCSR(Sparse):
    '''Jacobian caclulator using `scipy.sparse.csr_array()` matrices.'''

    name = 'sparse_csr'

    _Array = scipy.sparse.csr_array


class SparseCSC(Sparse):
    '''Jacobian caclulator using `scipy.sparse.csc_array()` matrices.'''

    name = 'sparse_csc'

    _Array = scipy.sparse.csc_array


class SparseBSR(Sparse):
    '''Jacobian caclulator using `scipy.sparse.bsr_array()` matrices.'''

    name = 'sparse_bsr'

    @staticmethod
    def _get_shape(arg1, *, shape=None, **_):
        '''Get the shape.'''
        if shape is None:
            try:
                # `arg1` has a `shape` attribute, e.g. a dense or
                # sparse array.
                shape = arg1.shape
            except AttributeError:
                # Use `scipy.sparse.bsr_array()` to get the shape
                # for other forms of `arg1`.
                shape = scipy.sparse.bsr_array(arg1).shape
            else:
                assert len(shape) == 2
        return shape

    def _get_blocksize(self, arg1, *, blocksize=None, **kwds):
        '''Get the block size with an informed guess when it is not
        passed explicitly.'''
        if blocksize is None:
            if isinstance(self.model, age_structured.Model):
                # Use the number of age groups for the dimensions of the
                # blocksize.
                blocksize = (len(self.model.a), ) * 2
                # Handle (m, 1) and (1, n) vectors.
                shape = self._get_shape(arg1, **kwds)
                blocksize = tuple(
                    1 if sh == 1 else bs
                    for (sh, bs) in zip(shape, blocksize)
                )
            # Otherwise keep `blocksize = None` so that
            # `scipy.sparse.bsr_array()` chooses `blocksize`.
        return blocksize

    def _Array(self, arg1, **kwds):  # pylint: disable=invalid-name
        '''`scipy.sparse.bsr_array()` with an informed guess for
        `blocksize` when it is not passed explicitly.'''
        blocksize = self._get_blocksize(arg1, **kwds)
        return scipy.sparse.bsr_array(arg1, blocksize=blocksize, **kwds)


def _get_subclasses(cls):
    '''Recursively get all of subclasses of `cls`.'''
    subs = (_get_subclasses(sub)
            for sub in cls.__subclasses__())
    return sum(subs, [cls])


def _get_calculators():
    '''Get the Jacobian calculators by name.'''
    calculators = {cls.name: cls
                   for cls in _get_subclasses(Base)}
    return calculators


def Calculator(solver, method):  # pylint: disable=invalid-name
    '''Factory function to build a Jacobian calculator.'''
    calculators = _get_calculators()
    try:
        calculator = calculators[method]
    except KeyError:
        raise ValueError(f'{method=}') from None
    return calculator(solver)
