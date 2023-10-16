'''Linear-algebra solver.'''

import scipy.linalg
import scipy.sparse


def solve(arr, vec, overwrite_a=False, overwrite_b=False):
    '''Solve the matrix system A @ x = b.'''
    if scipy.sparse.issparse(arr):
        if scipy.sparse.issparse(vec):
            arr = arr.tocsc()
            vec = vec.tocsc()
        result = scipy.sparse.linalg.spsolve(arr, vec)
    else:
        result = scipy.linalg.solve(arr, vec,
                                    overwrite_a=overwrite_a,
                                    overwrite_b=overwrite_b)
    return result
