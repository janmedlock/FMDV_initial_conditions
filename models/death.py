'''Death rate.'''

import numpy
import pandas


# Annual survival in age bands (ages in years).
_annual_survival = {(0, 1): 0.66,
                    (1, 3): 0.79,
                    (3, 12): 0.88,
                    (12, numpy.inf): 0.66}
_annual_survival = pandas.Series(
    _annual_survival.values(),
    index=pandas.IntervalIndex.from_tuples(_annual_survival.keys(),
                                           closed='left')
)

# Death rate with units per year.
_death_rate = - numpy.log(_annual_survival)


def rate(age):
    '''Death rate.'''
    out = _death_rate[age]
    try:
        out.set_axis(age, inplace=True)
    except AttributeError:
        pass
    return out
