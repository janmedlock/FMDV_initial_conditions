'''Helpers for scipy.optimize().'''

import scipy.optimize


_METHOD_DEFAULT = 'hybr'
_METHODS_WITHOUT_DISP_OPTION = ('hybr', 'lm')


def root(*args, sparse=False, display=False, **kwds):
    '''`scipy.optimize.root()` with improved default values for its
    'method' and 'options' arguments.'''
    if sparse:
        # Default to `root(..., method='krylov', ...)' if not set in `kwds`.
        kwds = {'method': 'krylov'} | kwds
    # Set the 'disp' option from `display` for some methods.
    method = kwds.get('method', _METHOD_DEFAULT)
    if method not in _METHODS_WITHOUT_DISP_OPTION:
        # Set `kwds['options']['disp'] = display` if not set in `kwds.`
        kwds = {'options': {'disp': display}} | kwds
    return scipy.optimize.root(*args, **kwds)
