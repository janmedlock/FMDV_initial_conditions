'''Statistical helper functions.'''

import numpy


def hazard(rv, z):
    '''The hazard for the random variable `rv` at time `z`.'''
    # `rv.pdf(z) / rv.sf(z)`, but handle using logs for better over- &
    # under-flow behavior.
    loghazard = rv.logpdf(z) - rv.logsf(z)
    return numpy.exp(loghazard)
