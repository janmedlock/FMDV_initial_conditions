'''Helpers for scipy.optimize().'''

import scipy.optimize


_method_default = 'hybr'
_methods_without_disp_option = ('hybr', 'lm')


def root(*args, sparse=False, display=False, **kwds):
    if sparse:
        # Default to `root(..., method='krylov', ...)' if not set in `kwds`.
        kwds = dict(method='krylov') | kwds
    # Set the 'disp' option from `display` for some methods.
    method = kwds.get('method', _method_default)
    if method not in _methods_without_disp_option:
        # Set `kwds['options']['disp'] = display` if not set in `kwds.`
        kwds = dict(options=dict(disp=display)) | kwds
    return scipy.optimize.root(*args, **kwds)
