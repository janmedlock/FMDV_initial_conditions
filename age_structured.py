#!/usr/bin/python3
'''Based on our FMDV work, this is an age-structured SIR model with
periodic birth rate.'''

import dataclasses

import numpy
import pandas

import solvers
import solvers.age_structured


_annual_survival = pandas.Series({(0, 1): 0.66,
                                  (1, 3): 0.79,
                                  (3, 12): 0.88,
                                  (12, numpy.inf): 0.66})
_idx = pandas.IntervalIndex.from_tuples(_annual_survival.index,
                                        closed='left')
_annual_survival.set_axis(_idx, inplace=True)
_death_rate = - numpy.log(_annual_survival)


@dataclasses.dataclass
class Parameters:
    '''Model parameters.'''
    birth_rate_variation: float = 0.5
    transmission_rate: float = 2.8 * 365  # per year
    recovery_rate: float = 1 / 5.7 * 365  # per year

    @staticmethod
    def death_rate(age):
        return _death_rate[age]

    @staticmethod
    def maternity_rate(age):
        return numpy.where(age < 4, 0, 1)


class Model:
    '''Age-structured SIR model.'''

    STATES = ('susceptible', 'infectious', 'recovered')

    PERIOD = 1  # year

    def __init__(self, **kwds):
        self.parameters = Parameters(**kwds)
        self._set_birth_rate_mean()

    def birth_rate(self, t):
        '''Periodic birth rate.'''
        return (self.birth_rate_mean
                * (1 + (self.parameters.birth_rate_variation
                        * numpy.sqrt(2)
                        * numpy.cos(2 * numpy.pi * t / self.PERIOD))))

    def _set_birth_rate_mean(self):
        '''Set `birth_rate_mean` to the value that gives zero
        population growth rate.'''
        self.birth_rate_mean = 0.5  # Guess.
        scale = solvers.age_structured.get_birth_scaling_for_no_pop_growth(
            self.parameters.death_rate, self.parameters.maternity_rate,
            self.birth_rate, self.PERIOD
        )
        self.birth_rate_mean *= scale


class ModelConstantBirth(Model):
    '''The SIR model with constant birth rate.'''

    def __init__(self, **kwds):
        super().__init__(birth_rate_variation=0, **kwds)


if __name__ == '__main__':
    model = Model()
