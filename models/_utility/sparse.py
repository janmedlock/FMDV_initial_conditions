'''Tools for sparse matrices.'''

import numpy
import scipy.sparse


# The sparse array format to use throughout.
array = scipy.sparse.csr_array


def density(A):
    '''Get the density of `A`.'''
    (nrow, ncol) = A.shape
    dnst = A.nnz / nrow / ncol
    return dnst


def identity(n, *args, **kwds):
    '''Build a sparse `array()` identity matrix.'''
    I = scipy.sparse.identity(n, *args, **kwds)
    return array(I)


def block_diag(mats, *args, **kwds):
    '''Build a sparse `array()` from the block diagonals `mat`.'''
    arr = scipy.sparse.block_diag(mats, *args, **kwds)
    return array(arr)


def bmat(blocks, *args, **kwds):
    '''Build a sparse `array()` from the sparse sub-blocks `blocks`.'''
    arr = scipy.sparse.bmat(blocks, *args, **kwds)
    return array(arr)


def diags(diagonals, offsets=0, shape=None, *args, **kwds):
    '''Build a sparse `array()` from the `diagonals`.'''
    arr = scipy.sparse.diags(diagonals, offsets=offsets, shape=shape,
                             *args, **kwds)
    return array(arr)


def hstack(blocks, *args, **kwds):
    '''Build a sparse `array()` by horizontally stacking `blocks`.'''
    arr = scipy.sparse.hstack(blocks, *args, **kwds)
    return array(arr)


def vstack(blocks, *args, **kwds):
    '''Build a sparse `array()` by vertically stacking `blocks`.'''
    arr = scipy.sparse.vstack(blocks, *args, **kwds)
    return array(arr)


def _idx_convert(arg):
    try:
        return slice(*arg)
    except TypeError:
        return arg


def _loc_convert(loc):
    return tuple(map(_idx_convert, loc))


def _get_len(idx):
    # Get the end of any slices.
    idx = (x.stop if isinstance(x, slice) else x
           for x in idx)
    # Drop `None` values, which come from slices no `stop`.
    idx = (x for x in idx if x is not None)
    # The shape is the max or `None` is the list is empty.
    return max(idx, default=None)


def _get_shape(locs):
    # Split the row and column indices into separate lists.
    row_col = zip(*locs)
    shape = tuple(map(_get_len, row_col))
    if any(val is None for val in shape):
        raise ValueError('shape is undetermined.')
    return shape


def array_from_dict(data, shape=None, *args, **kwds):
    '''Build a sparse `array()` from the dictionary `data` with keys
    (row, col).'''
    locs = map(_loc_convert, data.keys())
    if shape is None:
        try:
            shape = _get_shape(locs)
        except ValueError as err:
            raise ValueError('Set the `shape` argument.') from err
    arr = scipy.sparse.dok_array(shape, *args, **kwds)
    for (loc, val) in zip(locs, data.values()):
        arr[loc] = val
    return array(arr)


def diags_from_dict(data, shape=None, *args, **kwds):
    '''Build a sparse `array()` from the dictionary of diagonals
    `data`.'''
    diagonals = data.values()
    offsets = tuple(data.keys())
    return diags(diagonals, offsets=offsets, shape=shape,
                 *args, **kwds)


def equals(a, b):
    '''Whether the sparse matrices `a` and `b` are equal.'''
    if a.shape != b.shape:
        return False
    err = a - b
    return numpy.allclose(err.data, 0)
