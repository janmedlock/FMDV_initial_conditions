'''Classes and subclasses.'''


def all_subclasses(cls):
    '''Find all subclasses, subsubclasses, etc. of `cls`.'''
    for c in cls.__subclasses__():
        yield c
        for s in all_subclasses(c):
            yield s
