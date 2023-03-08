'''Linear algebra.'''

import numpy
import scipy.linalg
import scipy.sparse


# Map the first letter of 'which' to whether to reverse the sort order.
_reverse = dict(L=True,
                S=False)

# Map the second letter of 'which' to a sort key function.
_key = dict(M=numpy.abs,
            R=numpy.real,
            I=numpy.imag)


def _parse_which(which):
    '''Parse `which` into a key function and whether to reverse the
    sort.'''
    try:
        (direction, kind) = which
        key = _key[kind]
        reverse = _reverse[direction]
    except Exception as err:
        raise ValueError(f'{which=}') from err
    if reverse:
        key_ = key
        key = lambda x: -1. * key_(x)
    return key


def _sort_eigs(result, k, which, return_eigenvectors):
    '''Sort `result` by the eigenvalues using `which` and only return
    the first `k`.'''
    if return_eigenvectors:
        (eigvals, eigvecs) = result
    else:
        eigvals = result
    key = _parse_which(which)
    # Only keep the first `k`.
    order = numpy.argsort(key(eigvals))[:k]
    eigvals_sorted = eigvals[order]
    if return_eigenvectors:
        eigvecs_sorted = eigvecs[:, order]
        result_sorted = (eigvals_sorted, eigvecs_sorted)
    else:
        result_sorted = eigvals_sorted
    return result_sorted


def eigs(A, k=5, which='LR', maxiter=10000, return_eigenvectors=True,
         *args, **kwds):
    '''Get the first `k` eigenvalues of `A`.'''
    if k < A.shape[0] - 1:
        result = scipy.sparse.linalg.eigs(
            A, k=k, which=which, maxiter=maxiter,
            return_eigenvectors=return_eigenvectors,
            *args, **kwds
        )
    else:
        # If A is smaller than (k+1) x (k+1), fall back to
        # `scipy.linalg.eig()`.
        result = scipy.linalg.eig(A, right=return_eigenvectors)
    return _sort_eigs(result, k, which, return_eigenvectors)


def get_dominant_eigen(A, which='LR', return_eigenvector=True,
                       maxiter=100000, *args, **kwds):
    '''Get the dominant eigenvalue & eigenvector of `A` using
    `scipy.sparse.linalg.eigs()`, which works for both sparse
    and dense matrices.
    `which='LR'` gets the eigenvalue with largest real part.
    `which='LM'` gets the eigenvalue with largest magnitude.'''
    # The solver just spins with inf/NaN entries.
    # I think this check handles dense & sparse matrices, etc.
    assert numpy.isfinite(A[numpy.nonzero(A)]).all(), \
        '`A` has inf/NaN entries.'
    result = eigs(A, k=1, which=which, maxiter=maxiter,
                  return_eigenvectors=return_eigenvector,
                  *args, **kwds)
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


def solve(A, b, overwrite_a=False, overwrite_b=False):
    '''Solve the matrix system A @ x = b.'''
    if scipy.sparse.issparse(A):
        return scipy.sparse.linalg.spsolve(A, b)
    else:
        return scipy.linalg.solve(A, b,
                                  overwrite_a=overwrite_a,
                                  overwrite_b=overwrite_b)


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
    # Copy `arr`, then set the diagonal to 0 and check whether the
    # rest are nonpositive.
    if scipy.sparse.issparse(arr):
        # Copy and convert to COO format, which is efficient for
        # changing sparsity and has a `.min()` method, which is used
        # in `is_nonpositive()`.
        arr = arr.tocoo(copy=True)
        arr.setdiag(0)
    else:
        arr = arr.copy()
        diag = numpy.diag_indices_from(arr)
        arr[diag] = 0
    return is_nonpositive(arr)


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
