'''Eigenvalue and eigenvector utilities.'''

import numpy
import scipy.linalg
import scipy.sparse

from . import _sort
from .. import linalg


def eigs(arr,
         k=5, which='LR', return_eigenvectors=True, maxiter=None,
         **kwds):
    '''Get the first `k` eigenvalues of `arr` and, optionally, their
    corresponding right eigenvectors. The eigenvalues are sorted by:
    `which='LM'` sorts by magnitude, largest first.
    `which='SM'` sorts by magnitude, smallest first.
    `which='LR'` sorts by real part, largest first.
    `which='SR'` sorts by real part, smallest first.
    `which='LI'` sorts by imaginary part, largest first.
    `which='SI'` sorts by imaginary part, smallest first.'''
    assert numpy.ndim(arr) == 2, '`arr` is not 2 dimensional.'
    size = arr.shape[0]
    if k < size - 1:
        # Use `scipy.sparse.linalg.eigs()` for efficiency.
        if maxiter is None:
            # The default `maxiter` in `scipy.sparse.linalg.eigs()` is
            # `size * 10`. Make sure it is at least 10,000.
            # TODO: Is this necessary?
            maxiter = max(size * 10, 10000)
        # The solver just spins when `arr` has inf or NaN entries, so
        # make sure `arr` is finite.
        assert linalg.is_finite(arr), '`arr` has inf or NaN entries.'
        result = scipy.sparse.linalg.eigs(
            arr, k=k, which=which,
            return_eigenvectors=return_eigenvectors,
            maxiter=maxiter,
            **kwds
        )
    else:
        # Fall back to `scipy.linalg.eig()`.
        if scipy.sparse.issparse(arr):
            arr = arr.toarray()
        result = scipy.linalg.eig(arr, right=return_eigenvectors)
    return _sort.eigs(result, k, which, return_eigenvectors)


def eig_dominant(arr,
                 which='LR', return_eigenvector=True,
                 **kwds):
    '''Get the dominant eigenvalue & eigenvector of `arr`.
    `which='LM'` gets the eigenvalue with largest magnitude.
    `which='LR'` gets the eigenvalue with largest real part.'''
    result = eigs(arr, k=1, which=which,
                  return_eigenvectors=return_eigenvector,
                  **kwds)
    if return_eigenvector:
        (eigvals, eigvecs) = result
    else:
        eigvals = result
    eigval_dom = numpy.real_if_close(eigvals[0])
    assert numpy.isreal(eigval_dom), \
        f'Complex dominant eigenvalue: {eigval_dom}'
    if return_eigenvector:
        eigvec_dom = eigvecs[:, 0]
        eigvec_dom = numpy.real_if_close(eigvec_dom / eigvec_dom.sum())
        assert all(numpy.isreal(eigvec_dom)), \
            f'Complex dominant eigenvector: {eigvec_dom}'
        assert all((numpy.real(eigvec_dom) >= 0)
                   | numpy.isclose(eigvec_dom, 0)), \
            f'Negative component in the dominant eigenvector: {eigvec_dom}'
        eigvec_dom = eigvec_dom.clip(0, numpy.PINF)
        result = (eigval_dom, eigvec_dom)
    else:
        result = eigval_dom
    return result


def condition_number(arr):
    '''Get the condition number of `arr`.'''
    eigval_max = eig_dominant(arr, which='LM', return_eigenvector=False)
    eigval_min = eig_dominant(arr, which='SM', return_eigenvector=False)
    cond_num = numpy.abs(eigval_max / eigval_min)
    return cond_num
