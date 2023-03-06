'''Disk cache to reuse slow computations.'''

import pathlib

import joblib


# The cache is in a subdirectory of the root module directory.
_depth = len(__name__.split('.')) - 1
assert _depth > 0
_root = pathlib.Path(__file__).parents[_depth - 1]
_path = _root / '_cache'

cache = joblib.Memory(_path, compress=True, verbose=1)
