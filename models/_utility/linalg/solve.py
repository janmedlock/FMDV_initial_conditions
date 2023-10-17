'''Linear-algebra solver.'''

import scipy.linalg
import scipy.sparse


def solve(a, b, overwrite_a=False, overwrite_b=False):
    '''Solve the matrix system A @ x = b.'''
    if scipy.sparse.issparse(a):
        if scipy.sparse.issparse(b):
            a = a.tocsc()
            b = b.tocsc()
        result = scipy.sparse.linalg.spsolve(a, b)
    else:
        result = scipy.linalg.solve(a, b,
                                    overwrite_a=overwrite_a,
                                    overwrite_b=overwrite_b)
    return result
