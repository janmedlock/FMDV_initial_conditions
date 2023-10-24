'''Miscellaneous utilities.'''


def call_to_str(func, *args, **kwds):
    '''Build a string of the function call.'''
    args_kwds_strs = []
    if len(args) > 0:
        args_str = ', '.join(map(str, args))
        args_kwds_strs.append(args_str)
    if len(kwds) > 0:
        kwds_str = ', '.join(f'{k}={v}' for (k, v) in kwds.items())
        args_kwds_strs.append(kwds_str)
    args_kwds_str = ', '.join(args_kwds_strs)
    return f'{func.__name__}({args_kwds_str})'
