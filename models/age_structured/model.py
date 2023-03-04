'''Based on our FMDV work, this is an age-structured model.'''

import numpy
import pandas

from . import _age, _population, _solver
from .. import parameters, unstructured, _model, _utility


class Model(parameters.AgeDependent,
            unstructured.Model):
    '''Age-structured model.'''

    _Solver = _solver.Solver

    def __init__(self, *args, a_max=_age.max_default, **kwds):
        super().__init__(*args, **kwds)
        self.a_step = self._Solver._get_a_step(self.t_step)
        self.a = _utility.numerical.build_t(0, a_max, self.a_step)

    def _build_index(self):
        '''Extend the `pandas.Index()` for solutions with the 'age'
        level.'''
        idx_other = super()._build_index()
        idx_age = pandas.Index(self.a, name='age')
        idx = pandas.MultiIndex.from_product([idx_other, idx_age])
        return idx

    def _build_weights(self):
        '''Adjust the weights for the 'age' level.'''
        weights_other = super()._build_weights()
        # Each 'age' has weight `self.a_step`.
        weights_age = self.a_step
        weights = weights_other * weights_age
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

    def build_initial_conditions(self, *args, **kwds):
        '''Adjust the initial conditions for the 'age' level.'''
        Y_other = super().build_initial_conditions()
        n_age = self.stable_age_density(*args, **kwds)
        y = Y_other * n_age
        return y
