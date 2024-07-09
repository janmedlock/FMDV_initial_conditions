'''Helpers for integrals over variables.'''


def get_level_values(obj, axis, level):
    idx = obj.axes[axis]
    vals = idx.get_level_values(level)
    return vals


def integral(obj, level, integral_group):
    '''Integrate `obj` over `level` using `integral_group()`.'''
    # Operate on the last axis.
    axis = obj.ndim - 1
    transpose = (axis == 1)
    if transpose:
        obj = obj.T
        axis = 0
    # Group by all the levels except `level`.
    others = obj.index.names.difference({level})
    grouper = obj.groupby(others, observed=False, dropna=False)
    agg = grouper.apply(integral_group, axis) \
                 .dropna()
    if transpose:
        agg = agg.T
    return agg
