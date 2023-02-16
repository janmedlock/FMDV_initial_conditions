'''Linear algebra.'''

import numpy
import scipy.sparse.linalg


def get_dominant_eigen(A, which='LR', return_eigenvector=True,
                       maxiter=100000, *args, **kwargs):
    '''Get the dominant eigenvalue & eigenvector of `A` using
    `scipy.sparse.linalg.eigs()`, which works for both sparse
    and dense matrices.
    `which='LR'` gets the eigenvalue with largest real part.
    `which='LM'` gets the eigenvalue with largest magnitude.'''
    # The solver just spins with inf/NaN entries.
    # I think this check handles dense & sparse matrices, etc.
    assert numpy.isfinite(A[numpy.nonzero(A)]).all(), \
        '`A` has inf/NaN entries.'
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
        v0 = v0.clip(0, numpy.PINF)
        return (l0, v0)
    else:
        return l0


def is_positive(arr):
    '''Check whether all entries are positive.'''
    # `(arr > 0).all()` doesn't work with sparse matrices.
    return arr.min() > 0


def is_nonnegative(arr):
    '''Check whether all entries are nonnegative.'''
    # `(arr >= 0).all()` doesn't work with sparse matrices.
    return arr.min() >= 0


def is_negative(arr):
    '''Check whether all entries are negative.'''
    return is_positive(-arr)


def is_nonpositive(arr):
    '''Check whether all entries are nonpositive.'''
    return is_nonnegative(-arr)


def is_Z_matrix(arr):
    '''Check whether `arr` is a Z-matrix.'''
    # Set the diagonal to 0 then check whether the rest are nonpositive.
    diag = arr.diagonal()
    if isinstance(arr, scipy.sparse.spmatrix):
        diag = scipy.sparse.diags(diag, format=arr.format)
    else:
        diag = numpy.diagflat(diag)
    offdiag = arr - diag
    return is_nonpositive(offdiag)


def is_Metzler_matrix(arr):
    '''Check whether `arr` is a Metzler matrix.'''
    return is_Z_matrix(-arr)


def is_M_matrix(arr):
    '''Check whether `arr` is an M-matrix.'''
    if is_Z_matrix(arr):
        eigval_min = get_dominant_eigen(arr, which='SR',
                                        return_eigenvector=False)
        return eigval_min.real >= 0
    else:
        return False


def is_nonsingular_M_matrix(arr):
    '''Check whether `arr` is a non-singular M-matrix.'''
    if is_Z_matrix(arr):
        eigval_min = eigval_extremal(arr, which='SR')
        return (eigval_min.real >= 0) and (eigval_min != 0)
    else:
        return False
