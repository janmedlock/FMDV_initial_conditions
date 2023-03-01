'''Based on our FMDV work, this is an age-structured model.'''

import numpy
import pandas

from . import _population, _solver
from .. import unstructured, _model
from .._utility import numerical


class Mixin:
    '''Attributes for models that have an age variable.'''

    def _build_index_state_age(self):
        '''Build the 'state' and 'age' levels of a `pandas.Index()`
        for solutions.'''
        idx_state = self._build_index_state()
        idx_age = pandas.Index(self.a, name='age')
        idx = pandas.MultiIndex.from_product([idx_state, idx_age])
        return idx

    def _build_weights_state_age(self):
        '''Build weights for the 'state' and 'age' levels.'''
        weights_state = self._build_weights_state()
        # Each 'age' has weight `self.a_step`.
        weights_age = self.a_step
        weights = weights_state * weights_age
        return weights

    def stable_age_density(self, *args, **kwds):
        '''Get the stable age density.'''
        (a, v_dom) = _population.stable_age_density(self.birth, self.death,
                                                    *args, **kwds)
        # Interpolate the logarithm of `v_dom` to `self.a`.
        assert numpy.all(v_dom > 0)
        logn = numpy.interp(self.a, a, numpy.log(v_dom))
        n = numpy.exp(logn)
        # Normalize to integrate to 1.
        n /= n.sum() * self.a_step
        idx_age = self._get_index_level('age')
        dens = pandas.Series(n, index=idx_age)
        return dens

    def _build_initial_conditions_state_age(self, *args, **kwds):
        '''Build the initial conditions for the 'state' and 'age' levels.'''
        Y_state = self._build_initial_conditions_state()
        n_age = self.stable_age_density(*args, **kwds)
        y = Y_state * n_age
        return y


class Model(_model.Model, unstructured.Mixin, Mixin):
    '''Age-structured model.'''

    _Solver = _solver.Solver

    def __init__(self, a_step=0.001, a_max=25, **kwds):
        self.a_step = a_step
        self.a = numerical.build_t(0, a_max, self.a_step)
        super().__init__(**kwds)

    def _build_index(self):
        '''Build a `pandas.Index()` for solutions.'''
        idx = self._build_index_state_age()
        return idx

    def _build_weights(self):
        '''Build weights for the state vector.'''
        weights = self._build_weights_state_age()
        return weights

    def build_initial_conditions(self, *args, **kwds):
        '''Build the initial conditions.'''
        y = self._build_initial_conditions_state_age(*args, **kwds)
        return y
