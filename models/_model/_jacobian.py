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


class Base(_crank_nicolson.Mixin):  # pylint: disable=too-few-public-methods
    '''Calculate the Jacobian using the matrices from `Solver()`.'''

    name = 'base'

    # Attributes of `solver` that a subclass might convert to a
    # different format.
    _array_attrs = ('I', 'A', 'B', 'beta', 'T')

    def __init__(self, solver_):
        self.model = solver_.model
        self.t_step = solver_.t_step
        # Memoize so that identical matrices are only converted once.
        self._convert_arr = _MemoizeByID(self._convert_arr)
        for attr in self._array_attrs:
            obj = getattr(solver_, attr)
            setattr(self, attr, self._convert_obj(obj))

    def _convert_arr(self, arr, out=None):  # pylint: disable=method-hidden
        '''No-op that subclasses can override to convert `arr` to the
        desired format.'''
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

    def _make_column_vector(self, y):
        '''Convert `y` with shape (n, ) to shape (n, 1).'''
        assert numpy.ndim(y) == 1
        return numpy.asarray(y)[:, None]

    def _M(self, q, y_q, b_mid, out=None):  # pylint: disable=invalid-name
        '''Calculate the matrix
        M_q = A_q ± t_step / 2 (b_mid * B
                                + beta @ y_q @ T_q
                                + T_q @ y_q @ beta).'''
        # pylint: disable-next=invalid-name,no-member
        A_q = self.A[q]
        # The linear algebra is easier if `y_q` has shape (n, 1)
        # instead of just (n, ).
        y_q = self._make_column_vector(y_q)
        # For models with age structure, `T @ y @ beta` is *much* less
        # sparse.
        # pylint: disable-next=invalid-name
        BT_q = (
            # pylint: disable=no-member
            b_mid * self.B
            + self.beta @ y_q * self.T[q]
            + self.T[q] @ y_q @ self.beta
        )
        # M_q = A_q ± self.t_step / 2 * BT_q
        return self._cn_op(q, A_q, BT_q,
                           out=out)

    def _D(self, t_cur, y_cur, y_new):  # pylint: disable=invalid-name
        '''Calculate the D matrix at `t_cur`, given `y_cur` and `y_new`.'''
        # Compute `D`, the derivative of `y_cur` with respect to `y_new`,
        # which is `M_new @ D = M_cur`.
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.parameters.birth.rate(t_mid)
        M_cur = self._M('cur', y_cur, b_mid)  # pylint: disable=invalid-name
        M_new = self._M('new', y_new, b_mid)  # pylint: disable=invalid-name
        # pylint: disable-next=invalid-name
        return _utility.linalg.solve(M_new, M_cur,
                                     overwrite_a=True,
                                     overwrite_b=True)

    def _J(self, D):  # pylint: disable=invalid-name
        '''Calculate the Jacobian
        J = 1 / t_step * (D - I).'''
        # pylint: disable-next=no-member
        return (D - self.I) / self.t_step

    def calculate(self, t_cur, y_cur, y_new):
        '''Calculate the Jacobian at `t_cur`, given `y_cur` and `y_new`.'''
        D = self._D(t_cur, y_cur, y_new)  # pylint: disable=invalid-name
        return self._J(D)


class Dense(Base):  # pylint: disable=too-few-public-methods
    '''Jacobian caclulator using dense `numpy.ndarray` matrices.'''

    name = 'dense'

    @staticmethod
    def _empty(shape, dtype=float):
        '''Get an empty array.'''
        arr = numpy.empty(shape, dtype=dtype)
        return arr

    def _convert_arr(self, arr, out=None):
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
        shape = self.I.shape  # pylint: disable=no-member
        self.temp = self._empty(shape)
        # pylint: disable-next=invalid-name
        self.M = {q: self._empty(shape)
                  for q in self._q_vals}

    def __init__(self, solver_):
        super().__init__(solver_)
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
        # pylint: disable-next=invalid-name,no-member
        Ty_q = numpy.dot(self.T[q], y_q,
                         out=out[0])
        # For models with age structure, `Ty_q @ beta` is *much* less
        # sparse.
        # pylint: disable-next=invalid-name,no-member
        Tybeta_q = numpy.outer(Ty_q, self.beta,
                               out=self.temp)
        # out[0] = Ty_q is free now.
        BT_q = Tybeta_q  # pylint: disable=invalid-name
        # self.temp = BT_q is *not* free now.
        # betayT_q = (beta @ y_q) * T_q
        # pylint: disable-next=no-member
        betay_q = numpy.inner(self.beta[0], y_q)
        # pylint: disable-next=invalid-name,no-member
        betayT_q = numpy.multiply(betay_q, self.T[q],
                                  out=out)
        BT_q += betayT_q  # pylint: disable=invalid-name
        # out = betayT_q is free now.
        # bB_mid = b_mid * B
        # pylint: disable-next=invalid-name,no-member
        bB_mid = numpy.multiply(b_mid, self.B,
                                out=out)
        BT_q += bB_mid  # pylint: disable=invalid-name
        # out = bB_mid is free now.
        A_q = self.A[q]  # pylint: disable=invalid-name,no-member
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
        J -= self.I  # pylint: disable=invalid-name,no-member
        # J = (D - I) / t_step
        J /= self.t_step  # pylint: disable=invalid-name
        return J


class DenseMemmap(Dense):  # pylint: disable=too-few-public-methods
    '''Jacobian caclulator using memmapped dense `numpy.ndarray`
    matrices.'''

    name = 'dense_memmap'

    @staticmethod
    def _empty(shape, dtype=float):
        '''Get an empty array.'''
        memmap = _utility.numerical.memmaptemp(shape=shape,
                                               dtype=dtype)
        return memmap

    def _convert_arr(self, arr, out=None):
        '''Convert `arr` to a memmapped dense `numpy.ndarray` matrix.'''
        if out is None:
            out = self._empty(numpy.shape(arr), dtype=numpy.dtype(arr))
        return super()._convert_arr(arr, out=out)


class Sparse(Base):  # pylint: disable=too-few-public-methods
    '''Jacobian caclulator using the default sparse matrices.'''

    name = 'sparse'

    _Array = _utility.sparse.Array

    def _convert_arr(self, arr, out=None):
        '''Convert `arr` to the desired sparse format.'''
        if out is not None:
            raise ValueError(f'{out=}')
        return self._Array(arr)

    def _make_column_vector(self, y):
        '''Convert `y` with shape (n, ) to shape (n, 1).'''
        return self._Array(super()._make_column_vector(y))


class SparseCSR(Sparse):  # pylint: disable=too-few-public-methods
    '''Jacobian caclulator using `scipy.sparse.csr_array()` matrices.'''

    name = 'sparse_csr'

    _Array = scipy.sparse.csr_array


class SparseCSC(Sparse):  # pylint: disable=too-few-public-methods
    '''Jacobian caclulator using `scipy.sparse.csc_array()` matrices.'''

    name = 'sparse_csc'

    _Array = scipy.sparse.csc_array


class SparseBSR(Sparse):  # pylint: disable=too-few-public-methods
    '''Jacobian caclulator using `scipy.sparse.bsr_array()` matrices.'''

    name = 'sparse_bsr'

    @staticmethod
    def _get_shape(arg1):
        '''Get the shape from `arg1`.'''
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

    def _guess_blocksize(self, arg1, shape):
        '''Try to guess `blocksize`.'''
        if isinstance(self.model, age_structured.Model):
            # Use the number of age groups for the dimensions of the
            # blocksize.
            blocksize = (len(self.model.a), ) * 2
            # Handle (m, 1) and (1, n) vectors.
            if shape is None:
                shape = self._get_shape(arg1)
            blocksize = tuple(
                1 if sh == 1 else bs
                for (sh, bs) in zip(shape, blocksize)
            )
        else:
            # Let `scipy.sparse.bsr_array()` choose `blocksize`.
            blocksize = None
        return blocksize

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments,invalid-name  # noqa: E501
    def _Array(self, arg1,
               shape=None, dtype=None, copy=False, blocksize=None):
        '''`scipy.sparse.bsr_array()` with an informed guess for
        `blocksize` when it is not passed explicitly.'''
        if blocksize is None:
            blocksize = self._guess_blocksize(arg1, shape)
        return scipy.sparse.bsr_array(arg1,
                                      shape=shape, dtype=dtype, copy=copy,
                                      blocksize=blocksize)


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


def Calculator(solver_, method):  # pylint: disable=invalid-name
    '''Factory function to build a Jacobian calculator.'''
    calculators = _get_calculators()
    try:
        calculator = calculators[method]
    except KeyError:
        raise ValueError(f'{method=}') from None
    return calculator(solver_)
