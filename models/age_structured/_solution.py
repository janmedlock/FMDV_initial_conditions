'''Class to hold solutions.'''

import numpy
import pandas


def Solution(y, ages, t=None, states=None):
    '''A solution.'''
    ages = pandas.Index(ages, name='ages')
    if states is not None:
        states = pandas.Index(states, name='state')
    if t is None:
        Y = numpy.reshape(y, (len(ages), len(states)))
        return pandas.DataFrame(Y, index=ages, columns=states)
    else:
        t = pandas.Index(t, name='time')
        columns = pandas.MultiIndex.from_product((states, ages))
        return pandas.DataFrame(y, index=t, columns=columns)
