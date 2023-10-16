'''Miscellaneous linear-algebra utilities.'''

import numpy
import scipy.sparse

from . import eigen


def is_finite(arr):
    '''Check whether `arr` has all entries finite (i.e. not infinite
    or NaN).'''
    if scipy.sparse.issparse(arr):
        result = numpy.isfinite(arr[numpy.nonzero(arr)]).all()
    else:
        result = numpy.isfinite(arr).all()
    return result


def is_positive(arr):
    '''Check whether `arr` has all entries positive.'''
    if scipy.sparse.issparse(arr):
        result = arr.min() > 0
    else:
        result = (arr > 0).all()
    return result


def is_nonnegative(arr):
    '''Check whether `arr` has all entries nonnegative.'''
    if scipy.sparse.issparse(arr):
        result = arr.min() >= 0
    else:
        result = (arr >= 0).all()
    return result


def is_negative(arr):
    '''Check whether `arr` has all entries negative.'''
    return is_positive(-arr)


def is_nonpositive(arr):
    '''Check whether `arr` has all entries nonpositive.'''
    return is_nonnegative(-arr)


def is_Z_matrix(arr):
    '''Check whether `arr` is a Z-matrix.'''
    # Check whether the off-diagonal entries are nonpositive.
    # Copy `arr`, make its diagonal entries nonpositive by setting
    # them to 0, and check whether the resulting array is nonpositive.
    if scipy.sparse.issparse(arr):
        # Copy and convert to COO format, which is efficient for
        # changing sparsity and has a `.min()` method, which is used
        # in `is_nonpositive()`.
        arr_offdiag = arr.tocoo(copy=True)
        arr_offdiag.setdiag(0)
    else:
        arr_offdiag = arr.copy()
        idx_diag = numpy.diag_indices_from(arr_offdiag)
        arr_offdiag[idx_diag] = 0
    return is_nonpositive(arr_offdiag)


def is_Metzler_matrix(arr):
    '''Check whether `arr` is a Metzler matrix.'''
    # Whether `arr` is a Metzler matrix
    # is equivalent to
    # whether `-arr` is a Z matrix.
    return is_Z_matrix(-arr)


def is_M_matrix(arr):
    '''Check whether `arr` is an M-matrix.'''
    # `arr` is an M matrix
    # if and only if
    # `arr` is a Z matrix and
    # its eigenvalues all have nonnegative real part.
    if is_Z_matrix(arr):
        eigval_min = eigen.eig_dominant(arr, which='SR',
                                        return_eigenvector=False)
        result = eigval_min.real >= 0
    else:
        result = False
    return result


def is_nonsingular_M_matrix(arr):
    '''Check whether `arr` is a non-singular M-matrix.'''
    # `arr` is a nonsingular M matrix
    # if and only if
    # `arr` is a Z matrix,
    # its eigenvalues all have nonnegative real part, and
    # none of its eigenvalues are 0.
    if is_Z_matrix(arr):
        eigval_min = eigen.eig_dominant(arr, which='SR',
                                        return_eigenvector=False)
        result = (eigval_min.real >= 0) and (eigval_min != 0)
    else:
        result = False
    return result
