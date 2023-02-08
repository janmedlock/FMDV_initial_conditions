'''Tools for sparse matrices.'''

import scipy.sparse


def diags(diags, *args, **kwds):
    '''Build a `scipy.sparse` matrix from the dictionary of diagonals
    `diags`.'''
    return scipy.sparse.diags(diags.values(),
                              tuple(diags.keys()),
                              *args,
                              **kwds)
