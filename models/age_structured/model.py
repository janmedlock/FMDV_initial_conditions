'''Based on our FMDV work, this is an age-structured model.'''

import numpy
import pandas

from . import _population, _solver
from .. import unstructured, _model


class Mixin(unstructured.Mixin):
    '''Attributes for models that have an age variable.'''

    # The default maximum age `a_max` for `Model()`. HOW IS IT CHOSEN?
    DEFAULT_A_MAX = 25

    @property
    def a_step(self):
        '''Get the age step.'''
        return self._solver.a_step

    @property
    def a(self):
        '''Get the age vector.'''
        return self._solver.a

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

    def _integral_over_a_group(self, obj, axis):
        '''Integrate one group over age.'''
        a = _model.integral.get_level_values(obj, axis, 'age')
        assert len(a) == len(self.a)
        return obj.sum(axis=axis) * self.a_step

    def integral_over_a(self, obj, *args, **kwds):
        '''Integrate `obj` over 'age'.'''
        if isinstance(obj, numpy.ndarray):
            assert len(obj) == len(self.a)
            return obj.sum(*args, **kwds) * self.a_step
        elif isinstance(obj, (pandas.Series, pandas.DataFrame)):
            return _model.integral.integral(obj, 'age',
                                            self._integral_over_a_group)
        else:
            raise NotImplementedError

    def stable_age_density(self, *args, **kwds):
        '''Get the stable age density.'''
        (a, v_dom) = _population.stable_age_density(self.birth, self.death,
                                                    *args, **kwds)
        # Interpolate the logarithm of `v_dom` to `self.a`.
        assert numpy.all(v_dom > 0)
        logn = numpy.interp(self.a, a, numpy.log(v_dom))
        n = numpy.exp(logn)
        # Normalize to integrate to 1.
        n /= self.integral_over_a(n)
        idx_age = self._get_index_level('age')
        dens = pandas.Series(n, index=idx_age)
        return dens

    def _build_initial_conditions_state_age(self, *args, **kwds):
        '''Build the initial conditions for the 'state' and 'age' levels.'''
        Y_state = self._build_initial_conditions_state()
        n_age = self.stable_age_density(*args, **kwds)
        y = Y_state * n_age
        return y


class Model(_model.Model, Mixin):
    '''Age-structured model.'''

    _Solver = _solver.Solver

    def __init__(self, a_max=Mixin.DEFAULT_A_MAX, **kwds):
        self.a_max = a_max
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
