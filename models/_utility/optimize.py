'''Helpers for scipy.optimize().'''

import scipy.optimize


def root(*args, sparse=False, **kwds):
    if sparse:
        # Default to `root(..., method='krylov', ...)' if not set in `kwds`.
        kwds = dict(method='krylov') | kwds
    return scipy.optimize.root(*args, **kwds)
