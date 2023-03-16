'''Jacobian calculators.'''

import functools

import numpy
import scipy.sparse

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

    # Attributes that a subclass might convert to a different format.
    _array_attrs = ('I', 'beta', 'H', 'F', 'T', 'B')

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

    def calculate(self, t_cur, y_cur, y_new):
        '''Calculate the Jacobian at `t_cur`, given `y_cur` and `y_new`.'''
        # Compute `D`, the derivative of `y_cur` with respect to `y_new`,
        # which is `M_new @ D = M_cur`.
        # The linear algebra is easier if `y_cur` and `y_new` have
        # shape (n, 1) instead of just (n, ).
        y_cur = self._make_column_vector(y_cur)
        y_new = self._make_column_vector(y_new)
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.parameters.birth.rate(t_mid)
        M_new = (
            self.H['new']
            - self.t_step / 2 * (self.F['new']
                                 + self.beta @ y_new * self.T['new']
                                 + self.T['new'] @ y_new @ self.beta
                                 + b_mid * self.B)
        )
        M_cur = (
            self.H['cur']
            + self.t_step / 2 * (self.F['cur']
                                 + self.beta @ y_cur * self.T['cur']
                                 + self.T['cur'] @ y_cur @ self.beta
                                 + b_mid * self.B)
        )
        D = _utility.linalg.solve(M_new, M_cur,
                                  overwrite_a=True,
                                  overwrite_b=True)
        J = (D - self.I) / self.t_step
        return J


class Dense(Base):
    '''Jacobian caclulator using dense `numpy.ndarray` matrices.'''

    _name = 'dense'

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


class DenseMemmap(Dense):
    '''Jacobian caclulator using memmapped dense `numpy.ndarray`
    matrices.'''

    _name = 'dense_memmap'

    @classmethod
    def _convert_arr(cls, arr):
        '''Convert `arr` to a memmapped dense `numpy.ndarray` matrix.'''
        memmap = _utility.numerical.memmaptemp(shape=numpy.shape(arr),
                                               dtype=numpy.dtype(arr))
        return super()._convert_arr(arr, out=memmap)


class Sparse(Base):
    '''Jacobian caclulator using the default sparse matrices.'''

    _name = 'sparse'

    _array = _utility.sparse.array

    # In `calculate()`, for models with age structure, `T @ y @ beta`
    # makes the `M` *much* less sparse.

    @classmethod
    def _convert_arr(cls, arr):
        '''Convert `arr` to the desired sparse format.'''
        return cls._array(arr)

    @classmethod
    def _make_column_vector(cls, y):
        '''Convert `y` with shape (n, ) to shape (n, 1).'''
        return cls._array(super()._make_column_vector(y))


class SparseCSR(Sparse):
    '''Jacobian caclulator using `scipy.sparse.csr_array()` matrices.'''

    _name = 'sparse_csr'

    _array = scipy.sparse.csr_array


class SparseCSC(Sparse):
    '''Jacobian caclulator using `scipy.sparse.csc_array()` matrices.'''

    _name = 'sparse_csc'

    _array = scipy.sparse.csc_array


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
        Calculator = calculators[method]
    except KeyError:
        raise ValueError(f'{method=}')
    else:
        return Calculator(solver)
