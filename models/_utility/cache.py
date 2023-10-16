'''Disk cache to reuse slow computations.'''

import pathlib

import joblib


# The name to use for the cache directory.
_DIR = '_cache'


def _cache():
    '''Build the cache.'''
    # The cache is in a subdirectory of the root module directory.
    depth = len(__name__.split('.')) - 1
    assert depth > 0
    root = pathlib.Path(__file__).parents[depth - 1]
    path = root / _DIR
    memory = joblib.Memory(path, compress=True, verbose=1)
    return memory.cache


cache = _cache()
