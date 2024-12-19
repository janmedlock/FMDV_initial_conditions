'''Based on our FMDV work, this is an age-structured model.'''

import functools

import numpy
import pandas

from . import _base, _solver
from .. import parameters, unstructured, _model


class Model(_base.Model, unstructured.Model):
    '''Age-structured model.'''

    _Parameters = parameters.ModelParametersAgeDependent

    _Solver = _solver.Solver

    def _build_index(self):
        '''Extend the `pandas.Index()` for solutions with the 'age'
        level.'''
        idx_other = super()._build_index()
        idx_age = pandas.Index(self.a, name='age')
        return pandas.MultiIndex.from_product([idx_other, idx_age])

    @functools.cached_property
    def _weights(self):
        '''Adjust the weights for the 'age' level.'''
        weights_other = super()._weights
        # Each 'age' has weight `self.a_step`.
        weights_age = self.a_step
        return weights_other * weights_age

    def _integral_over_a_group(self, obj, axis):
        '''Integrate one group over age.'''
        a = _model.integral.get_level_values(obj, axis, 'age')
        assert len(a) == len(self.a)
        return obj.sum(axis=axis) * self.a_step

    def integral_over_a(self, obj, *args, **kwds):
        '''Integrate `obj` over 'age'.'''
        if isinstance(obj, numpy.ndarray):
            assert len(obj) == len(self.a)
            val = obj.sum(*args, **kwds) * self.a_step
        elif isinstance(obj, (pandas.Series, pandas.DataFrame)):
            val = _model.integral.integral(obj, 'age',
                                           self._integral_over_a_group)
        else:
            raise NotImplementedError
        return val

    def stable_age_density(self, **kwds):
        '''Get the stable age density.'''
        (a, v_dom) = self.parameters.stable_age_density(**kwds)
        # Interpolate the logarithm of `v_dom` to `self.a`.
        assert numpy.all(v_dom > 0)
        logn = numpy.interp(self.a, a, numpy.log(v_dom))
        n = numpy.exp(logn)
        # Normalize to integrate to 1.
        n /= self.integral_over_a(n)
        idx_age = self._get_index_level('age')
        return pandas.Series(n, index=idx_age)

    def build_initial_conditions(self, **kwds):
        '''Adjust the initial conditions for the 'age' level.'''
        # pylint: disable-next=invalid-name
        Y_other = super().build_initial_conditions()
        n_age = self.stable_age_density(**kwds)
        return Y_other * n_age
