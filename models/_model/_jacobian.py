'''Jacobian calculators.'''

import abc
import functools

import numpy
import scipy.sparse

from .. import _utility


class _Base:
    '''Base for Jacobian caclulators.'''

    @property
    @abc.abstractmethod
    def _method(self):
        '''The name of the Jacobian method.'''

    # Attributes that a subclass might convert to a different format.
    _matrix_attrs = ('I', 'beta', 'H', 'F', 'T', 'B')

    def __init__(self, solver):
        self.model = solver.model
        self.t_step = solver.t_step
        self.I = solver.I
        self.beta = solver.beta
        self.H = solver.H
        self.F = solver.F
        self.T = solver.T
        self.B = solver.B

    @classmethod
    def _make_column_vector(cls, y):
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


class Base(_Base):
    '''Calculate the Jacobian using the solver matrices without
    conversion.'''

    _method = 'base'


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


class _Converter(_Base, metaclass=abc.ABCMeta):
    '''A Jacobian calculator that converts the matrices to a specific
    format.'''

    def __init__(self, solver):
        super().__init__(solver)
        # Memoize so that identical matrices are only converted once.
        self._convert = _MemoizeByID(self._convert)
        for attr in self._matrix_attrs:
            self._convert_attr(attr)

    @classmethod
    @abc.abstractmethod
    def _convert(cls, arr):
        '''Convert `arr` to the desired format.'''

    def _convert_obj(self, obj):
        '''Convert `obj` to the desired format.'''
        if isinstance(obj, (numpy.ndarray, scipy.sparse.spmatrix)):
            return self._convert(obj)
        elif isinstance(obj, dict):
            # Recursively convert dict values.
            return {key: self._convert_obj(val)
                    for (key, val) in obj.items()}
        else:
            raise TypeError

    def _convert_attr(self, attr):
        '''Convert the attribute `attr` to the solver sparse-matrix format.'''
        obj = getattr(self, attr)
        setattr(self, attr, self._convert_obj(obj))


class Dense(_Converter):
    '''Jacobian caclulator using dense `numpy.ndarray` matrices.'''

    _method = 'dense'

    @classmethod
    def _convert(cls, arr):
        '''Convert `obj` to a dense `numpy.ndarray` matrix.'''
        if isinstance(arr, numpy.ndarray):
            return arr
        elif isinstance(arr, scipy.sparse.spmatrix):
            return arr.toarray()
        else:
            raise TypeError


class _Sparse(_Converter, metaclass=abc.ABCMeta):
    '''Jacobian caclulator using `scipy.sparse` matrices.'''

    # In `calculate()`, for models with age structure, `T @ y @ beta`
    # makes the `M` *much* less sparse.

    @property
    @abc.abstractmethod
    def _array(self):
        '''The sparse-matrix type to use.'''

    @classmethod
    def _convert(cls, arr):
        '''Convert `arr` to the desired sparse format.'''
        return cls._array(arr)

    @classmethod
    def _make_column_vector(cls, y):
        '''Convert `y` with shape (n, ) to shape (n, 1).'''
        return cls._array(super()._make_column_vector(y))


class Sparse(_Sparse):
    '''Jacobian caclulator using the default sparse matrices.'''

    _method = 'sparse'

    _array = _utility.sparse.array


class SparseCSR(_Sparse):
    '''Jacobian caclulator using `scipy.sparse.csr_array()` matrices.'''

    _method = 'sparse_csr'

    _array = scipy.sparse.csr_array


class SparseCSC(_Sparse):
    '''Jacobian caclulator using `scipy.sparse.csc_array()` matrices.'''

    _method = 'sparse_csc'

    _array = scipy.sparse.csc_array


def _get_subclasses(cls):
    '''Recursively get all of subclasses of `cls`.'''
    subs = (_get_subclasses(sub)
            for sub in cls.__subclasses__())
    return sum(subs, [cls])


def _get_calculators():
    '''Get the Jacobian calculators by name.'''
    calculators = {}
    for cls in _get_subclasses(_Base):
        if isinstance(cls._method, str):
            calculators[cls._method] = cls
    return calculators


def Calculator(solver, method=None):
    '''Factory function to build a Jacobian calculator.'''
    if method is None:
        method = 'sparse_csc' if solver._sparse else 'base'
    calculators = _get_calculators()
    try:
        Calculator = calculators[method]
    except KeyError:
        raise ValueError(f'{method=}')
    else:
        return Calculator(solver)
