'''Sort eigenvalues and eigenvectors.'''

import numpy


# Map the first letter of `which` to whether to reverse the sort order.
_REVERSE = {
    'L': True,        # Largest
    'S': False,       # Smallest
}

# Map the second letter of `which` to a sort key function.
_KEY = {
    'M': numpy.abs,   # Magnitude
    'R': numpy.real,  # Real part
    'I': numpy.imag,  # Imaginary part
}


def _reverse(key):
    '''Reverse the sort order produced by `key`.'''
    def key_reversed(obj):
        return -1. * key(obj)
    return key_reversed


def _get_key(which):
    '''Parse `which` into a key function.'''
    try:
        (direction, kind) = which
        reverse = _REVERSE[direction]
        key = _KEY[kind]
    except Exception as err:
        raise ValueError(f'{which=}') from err
    if reverse:
        key = _reverse(key)
    return key


def eigs(result, k, which, return_eigenvectors):
    '''Sort `result` by the eigenvalues using `which` and only return
    the first `k`.'''
    if return_eigenvectors:
        (eigvals, eigvecs) = result
    else:
        eigvals = result
    key = _get_key(which)
    # Only keep the first `k`.
    order = numpy.argsort(key(eigvals))[:k]
    eigvals_sorted = eigvals[order]
    if return_eigenvectors:
        eigvecs_sorted = eigvecs[:, order]
        result_sorted = (eigvals_sorted, eigvecs_sorted)
    else:
        result_sorted = eigvals_sorted
    return result_sorted
