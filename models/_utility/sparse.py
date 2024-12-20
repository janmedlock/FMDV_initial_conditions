'''Tools for sparse matrices.'''

import numpy
import scipy.sparse


# The sparse array format to use throughout.
Array = scipy.sparse.csr_array


def density(arr):
    '''Get the density of `arr`.'''
    (nrow, ncol) = arr.shape
    dnst = arr.nnz / nrow / ncol
    return dnst


def identity(size, *args, **kwds):
    '''Build a sparse `Array()` identity matrix.'''
    eye = scipy.sparse.identity(size, *args, **kwds)
    return Array(eye)


def _fix_scalar_blocks(blocks):
    '''Convert any scalar blocks to 1x1 arrays.'''
    return [
        numpy.array([[block]]) if numpy.isscalar(block) else block
        for block in blocks
    ]


def block_diag(blocks, *args, **kwds):
    '''Build a sparse `Array()` from the block diagonals `blocks`.'''
    blocks = _fix_scalar_blocks(blocks)
    arr = scipy.sparse.block_diag(blocks, *args, **kwds)
    return Array(arr)


def bmat(blocks, *args, **kwds):
    '''Build a sparse `Array()` from the sparse sub-blocks `blocks`.'''
    blocks = [_fix_scalar_blocks(row) for row in blocks]
    arr = scipy.sparse.bmat(blocks, *args, **kwds)
    return Array(arr)


def diags(diagonals, offsets=0, shape=None, **kwds):
    '''Build a sparse `Array()` from the `diagonals`.'''
    arr = scipy.sparse.diags(diagonals, offsets=offsets, shape=shape, **kwds)
    return Array(arr)


def hstack(blocks, *args, **kwds):
    '''Build a sparse `Array()` by horizontally stacking `blocks`.'''
    blocks = _fix_scalar_blocks(blocks)
    arr = scipy.sparse.hstack(blocks, *args, **kwds)
    return Array(arr)


def vstack(blocks, *args, **kwds):
    '''Build a sparse `Array()` by vertically stacking `blocks`.'''
    blocks = _fix_scalar_blocks(blocks)
    arr = scipy.sparse.vstack(blocks, *args, **kwds)
    return Array(arr)


def kron(a, b, *args, **kwds):
    '''The Kronecker product.'''
    arr = scipy.sparse.kron(a, b, *args, **kwds)
    return Array(arr)


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


def array_from_dict(data, shape=None, **kwds):
    '''Build a sparse `Array()` from the dictionary `data` with keys
    (row, col).'''
    locs = map(_loc_convert, data.keys())
    if shape is None:
        try:
            shape = _get_shape(locs)
        except ValueError as err:
            raise ValueError('Set the `shape` argument.') from err
    arr = scipy.sparse.dok_array(shape, **kwds)
    for (loc, val) in zip(locs, data.values()):
        arr[loc] = val
    return Array(arr)


def diags_from_dict(data, shape=None, **kwds):
    '''Build a sparse `Array()` from the dictionary of diagonals
    `data`.'''
    diagonals = data.values()
    offsets = tuple(data.keys())
    return diags(diagonals, offsets=offsets, shape=shape, **kwds)


def equals(a, b):
    '''Whether the sparse matrices `a` and `b` are equal.'''
    if a.shape != b.shape:
        return False
    err = a - b
    return numpy.allclose(err.data, 0)
