'''Jacobian calculators.'''

import functools

import numpy
import scipy.sparse

from .. import age_structured
from .. import _utility


class _MemoizeByID:
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


class Base:
    '''Calculate the Jacobian using the matrices from `Solver()`.'''

    _name = 'base'

    # Attributes of `solver` that a subclass might convert to a
    # different format.
    _array_attrs = ('I', 'beta', 'A', 'T', 'B')

    def __init__(self, solver):
        self.model = solver.model
        self.t_step = solver.t_step
        # Memoize so that identical matrices are only converted once.
        self._convert_arr = _MemoizeByID(self._convert_arr)
        for attr in self._array_attrs:
            obj = getattr(solver, attr)
            setattr(self, attr, self._convert_obj(obj))

    @staticmethod
    def _convert_arr(arr):
        '''No-op that subclasses can override to convert `arr` to the
        desired format.'''
        return arr

    def _convert_obj(self, obj):
        '''Convert `obj` to the desired format. `obj` can be a
        `numpy.ndarray()` or a `scipy.sparse.spmatrix()`, or a
        `dict()` whose values are either of those.'''
        if isinstance(obj, (numpy.ndarray, scipy.sparse.spmatrix)):
            return self._convert_arr(obj)
        elif isinstance(obj, dict):
            # Recursively convert dict values.
            return {key: self._convert_obj(val)
                    for (key, val) in obj.items()}
        else:
            raise TypeError

    @staticmethod
    def _make_column_vector(y):
        '''Convert `y` with shape (n, ) to shape (n, 1).'''
        assert numpy.ndim(y) == 1
        return numpy.asarray(y)[:, None]

    def _M(self, q, y_q, b_mid):
        '''Calculate the M_q matrices.'''
        A = self.A[q]
        # The linear algebra is easier if `y_q` has shape (n, 1)
        # instead of just (n, ).
        y_q = self._make_column_vector(y_q)
        # For models with age structure, `T @ y @ beta` is *much* less
        # sparse.
        T_B = self.t_step / 2 * (self.beta @ y_q * self.T[q]
                                 + self.T[q] @ y_q @ self.beta
                                 + b_mid * self.B)
        # M = A ± T_B
        if q == 'cur':
            M = A + T_B
        elif q == 'new':
            M = A - T_B
        else:
            raise ValueError(f'{q=}')
        return M

    def calculate(self, t_cur, y_cur, y_new):
        '''Calculate the Jacobian at `t_cur`, given `y_cur` and `y_new`.'''
        # Compute `D`, the derivative of `y_cur` with respect to `y_new`,
        # which is `M_new @ D = M_cur`.
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.parameters.birth.rate(t_mid)
        M_cur = self._M('cur', y_cur, b_mid)
        M_new = self._M('new', y_new, b_mid)
        D = _utility.linalg.solve(M_new, M_cur,
                                  overwrite_a=True,
                                  overwrite_b=True)
        J = (D - self.I) / self.t_step
        return J


class Dense(Base):
    '''Jacobian caclulator using dense `numpy.ndarray` matrices.'''

    _name = 'dense'

    @staticmethod
    def _empty(shape, dtype=float):
        '''Get an empty array.'''
        arr = numpy.empty(shape, dtype=dtype)
        return arr

    @staticmethod
    def _convert_arr(arr, out=None):
        '''Convert `arr` to a dense `numpy.ndarray` matrix.'''
        if isinstance(arr, numpy.ndarray):
            if out is not None:
                out[:] = arr[:]
            return arr
        elif isinstance(arr, scipy.sparse.spmatrix):
            return arr.toarray(out=out)
        else:
            raise TypeError

    def _init_temp(self):
        '''Initialize temporary storage.'''
        shape = self.I.shape
        self.temp = [self._empty(shape)
                     for _ in range(2)]

    def __init__(self, solver):
        super().__init__(solver)
        self._init_temp()

    def _M(self, q, y_q, b_mid):
        '''Calculate the M_q matrices.'''
        # Use temporary storage to calculate
        # T_B = self.t_step / 2 * (self.beta @ y_q * self.T[q]
        #                          + self.T[q] @ y_q @ self.beta
        #                          + b_mid * self.B)
        #
        # T_y_beta = (T @ y) @ beta
        T_y = numpy.dot(self.T[q], y_q,
                        out=self.temp[0][0])
        # For models with age structure, `T_y @ beta` is *much* less
        # sparse.
        T_y_beta = numpy.outer(T_y, self.beta,
                               out=self.temp[1])
        # beta_y_T = (beta @ y) * T
        beta_y = numpy.inner(self.beta[0], y_q)
        beta_y_T = numpy.multiply(beta_y, self.T[q],
                                  out=self.temp[1])
        F_T_B += beta_y_T
        # b_B = b_mid * B
        b_B = numpy.multiply(b_mid, self.B,
                             out=self.temp[1])
        # T_B = T_y_beta
        T_B = numpy.add(T_y_beta, b_B,
                        out=self.temp[0])
        T_B *= self.t_step / 2
        A = self.A[q]
        # M = A ± T_B
        if q == 'cur':
            opp = numpy.add
        elif q == 'new':
            opp = numpy.subtract
        else:
            raise ValueError(f'{q=}')
        M = opp(self.A[q], T_B,
                out=self.temp[1])
        return M


class DenseMemmap(Dense):
    '''Jacobian caclulator using memmapped dense `numpy.ndarray`
    matrices.'''

    _name = 'dense_memmap'

    @staticmethod
    def _empty(shape, dtype=float):
        '''Get an empty array.'''
        memmap = _utility.numerical.memmaptemp(shape=shape,
                                               dtype=dtype)
        return memmap

    @classmethod
    def _convert_arr(cls, arr):
        '''Convert `arr` to a memmapped dense `numpy.ndarray` matrix.'''
        memmap = cls._empty(numpy.shape(arr), dtype=numpy.dtype(arr))
        return super()._convert_arr(arr, out=memmap)


class Sparse(Base):
    '''Jacobian caclulator using the default sparse matrices.'''

    _name = 'sparse'

    _Array = _utility.sparse.Array

    def _convert_arr(self, arr):
        '''Convert `arr` to the desired sparse format.'''
        return self._Array(arr)

    def _make_column_vector(self, y):
        '''Convert `y` with shape (n, ) to shape (n, 1).'''
        return self._Array(super()._make_column_vector(y))


class SparseCSR(Sparse):
    '''Jacobian caclulator using `scipy.sparse.csr_array()` matrices.'''

    _name = 'sparse_csr'

    _Array = scipy.sparse.csr_array


class SparseCSC(Sparse):
    '''Jacobian caclulator using `scipy.sparse.csc_array()` matrices.'''

    _name = 'sparse_csc'

    _Array = scipy.sparse.csc_array


class SparseBSR(Sparse):
    '''Jacobian caclulator using `scipy.sparse.bsr_array()` matrices.'''

    _name = 'sparse_bsr'

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
            n_ages = len(self.model.a)
            blocksize = (n_ages, n_ages)
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
    calculators = {cls._name: cls
                   for cls in _get_subclasses(Base)}
    return calculators


def Calculator(solver, method):
    '''Factory function to build a Jacobian calculator.'''
    calculators = _get_calculators()
    try:
        calculator = calculators[method]
    except KeyError:
        raise ValueError(f'{method=}') from None
    return calculator(solver)
