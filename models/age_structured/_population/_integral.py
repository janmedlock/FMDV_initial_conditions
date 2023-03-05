'''Integrals.'''


def over_a(arr, a_step, *args, **kwds):
    '''Integrate `arr` over age using `a_step` as the step
     size. `args` and `kwds` are passed on to `.sum()`.'''
    return arr.sum(*args, **kwds) * a_step
