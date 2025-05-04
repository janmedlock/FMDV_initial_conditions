'''Helpers for scipy.optimize().'''

import inspect

import numpy
import scipy.optimize


# Get the default `method` used in `scipy.optimize.root()`.
_METHOD_DEFAULT = inspect.signature(scipy.optimize.root) \
                         .parameters['method'] \
                         .default

# These methods do not use `options['disp']`.
_METHODS_WITHOUT_DISP_OPTION = {'hybr', 'lm'}


def root(*args, sparse=False, display=False, **kwds):
    '''`scipy.optimize.root()` with improved default for its 'options'
    argument that checks `result.success` and returns `result.x`.'''
    if sparse:
        # Default to `method = 'krylov'`.
        kwds = {'method': 'krylov'} | kwds
    # Set the 'disp' option from `display` for some methods.
    method = kwds.get('method', _METHOD_DEFAULT)
    if method not in _METHODS_WITHOUT_DISP_OPTION:
        # Default to `options['disp'] = display`.
        kwds['options'] = {'disp': display} | kwds.get('options', {})
    # Ignore some spurious warnings.
    with numpy.errstate(invalid='ignore'):
        result = scipy.optimize.root(*args, **kwds)
    assert result.success, result
    return result.x


def fixed_point(*args, maxiter=10000, **kwds):
    '''`scipy.optimize.fixed_point()` with improved default for its
    'maxiter' argument.'''
    return scipy.optimize.fixed_point(*args, maxiter=maxiter, **kwds)
