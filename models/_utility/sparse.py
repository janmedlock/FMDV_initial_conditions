'''Tools for sparse matrices.'''

import scipy.sparse


def diags(diags, *args, **kwds):
    '''Build a `scipy.sparse` matrix from the dictionary of diagonals
    `diags`.'''
    return scipy.sparse.diags(diags.values(),
                              tuple(diags.keys()),
                              *args,
                              **kwds)


def equals(a, b):
    '''Whether the sparse matrices `a` and `b` are equal.'''
    return (a.shape == b.shape) & ((a != b).nnz == 0)
