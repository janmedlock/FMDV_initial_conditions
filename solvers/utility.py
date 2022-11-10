'''Utilities.'''

import jax
import numpy
import scipy.sparse.linalg


def all_subclasses(cls):
    '''Find all subclasses, subsubclasses, etc. of `cls`.'''
    for c in cls.__subclasses__():
        yield c
        for s in all_subclasses(c):
            yield s


def arange(start, stop, step, endpoint=True, dtype=None):
    '''Like `numpy.arange()` but, if `endpoint` is True, ensure that
    `stop` is the output.'''
    arr = numpy.arange(start, stop, step, dtype=dtype)
    if endpoint and arr[-1] != stop:
        arr = numpy.hstack((arr, stop))
    return arr


def sort_by_real_part(arr):
    '''Sort the elements of `arr` by real part.'''
    order = arr.real.argsort()
    return arr[order]


def assert_nonnegative(y):
    '''Check that `y` is non-negative.'''
    assert (y >= 0).all(axis=None)


def get_dominant_eigen(A, which='LR', return_eigenvector=True,
                       maxiter=100000, *args, **kwargs):
    '''Get the dominant eigenvalue & eigenvector of `A` using
    `scipy.sparse.linalg.eigs()`, which works for both sparse
    and dense matrices.
    `which='LR'` gets the eigenvalue with largest real part.
    `which='LM'` gets the eigenvalue with largest magnitude.'''
    # The solver just spins with inf/NaN entries.
    # I think this check handles dense & sparse matrices, etc.
    assert numpy.isfinite(A[numpy.nonzero(A)]).all(), 'A has inf/NaN entries.'
    result = scipy.sparse.linalg.eigs(A, k=1, which=which, maxiter=maxiter,
                                      return_eigenvectors=return_eigenvector,
                                      *args, **kwargs)
    if return_eigenvector:
        (L, V) = result
    else:
        L = result
    l0 = numpy.real_if_close(L[0])
    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)
    if return_eigenvector:
        v0 = V[:, 0]
        v0 = numpy.real_if_close(v0 / v0.sum())
        assert all(numpy.isreal(v0)), \
            'Complex dominant eigenvector: {}'.format(v0)
        assert all((numpy.real(v0) >= 0) | numpy.isclose(v0, 0)), \
            'Negative component in the dominant eigenvector: {}'.format(v0)
        v0 = v0.clip(0, numpy.inf)
        return (l0, v0)
    else:
        return l0


def jacobian(func):
    '''Get the Jacobian matrix for the vector-valued `func`.'''
    jacfwd = jax.jacfwd(func, argnums=1)
    def jac(t, x):
        return numpy.stack(jacfwd(t, numpy.asarray(x)))
    return jac


def jacobian_matrix_product(func):
    '''Get the function that returns the product of the Jacobian of
    `func` at `t`, `x` with the matrix `M`.'''
    def jmp(t, x, M):
        def f(x):
            return func(t, x)
        def jvp(m):
            (_, jvp) = jax.jvp(f, (numpy.asarray(x), ), (m, ))
            return jvp
        return numpy.stack(jax.vmap(jvp)(M))
    return jmp


class TransformConstantSum:
    '''Reduce the dimension of `y` by 1 using its sum.'''

    def __init__(self, y):
        self.y_sum = y.sum()

    @staticmethod
    def __call__(y):
        '''Reduce the dimension of `y`.'''
        return y[:-1]

    def inverse(self, x):
        '''Expand the dimension of `x`.'''
        return numpy.hstack((x, self.y_sum - x.sum()))
