'''Data structures for lazy evaluation.'''

import collections


_FuncArgsKwds = collections.namedtuple(
    '_FuncArgsKwds',
    ('func', 'args', 'kwds'),
    defaults=((), {})  # args=(), kwds={}
)


class Dict(dict):
    '''Lazily evaluated dictionary.'''

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        for (key, value) in self.items():
            self[key] = _FuncArgsKwds(*value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, _FuncArgsKwds):
            value = value.func(*value.args, **value.kwds)
            super().__setitem__(key, value)
        return value
