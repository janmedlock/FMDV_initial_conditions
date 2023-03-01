'''Helers for integrals over variables.'''


def get_level_values(obj, axis, level):
    idx = obj.axes[axis]
    vals = idx.get_level_values(level)
    return vals


def integral(obj, level, integral_group):
    '''Integrate `obj` over `level` using `integral_group()`.'''
    # Operate on the last axis.
    axis = obj.ndim - 1
    # Group by all the levels on `axis` except `level`.
    others = obj.axes[axis].names.difference({level})
    grouper = obj.groupby(others, axis=axis, dropna=False)
    agg = grouper.apply(integral_group, axis) \
                 .dropna()
    return agg
