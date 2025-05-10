#!/usr/bin/python3
'''Test integrals.'''

import inspect
import operator

import pandas.testing
import pytest

from context import models


INTEGRAL = {
    'time_since_entry_structured': operator.attrgetter('integral_over_z'),
    'age_structured': operator.attrgetter('integral_over_a'),
    'combination': operator.attrgetter('integral_over_a_and_z'),
}


HOW_VALUES = (
    'survival',
    'all_in_first',
)


MODEL_KWS = {
    # Keep vectors short for fast testing ...
    't_step': 1e-1,
    # ... which requires turning off matrix checks.
    'solver_kwds': {
        '_check_matrices': False,
    }
}


def model_class(model_name):
    return getattr(models, model_name).Model


def build_model(model_name):
    return model_class(model_name)(**MODEL_KWS)


def build_vector(model):
    return model.build_initial_conditions


@pytest.fixture(scope='module')
def vector_unstructured():
    model = build_model('unstructured')
    return build_vector(model)()


def has_how(model_name):
    cls = model_class(model_name)
    method = build_vector(cls)
    sig = inspect.signature(method)
    return 'how' in sig.parameters


def build_hows(model_name):
    if has_how(model_name):
        return HOW_VALUES
    return (None, )


model_names_and_hows = [
    (model_name, how)
    for model_name in INTEGRAL.keys()
    for how in build_hows(model_name)
]


@pytest.mark.parametrize('model_name, how',
                         model_names_and_hows,
                         scope='class')
class TestIntegral:
    '''Test integrals.'''
    @pytest.fixture(scope='class')
    def model(self, model_name):
        return build_model(model_name)

    @pytest.fixture
    def integral(self, model_name, model, how):
        kws = {'how': how} if how is not None else {}
        vector = build_vector(model)(**kws)
        return INTEGRAL[model_name](model)(vector)

    def test_integral(self, integral, vector_unstructured):
        pandas.testing.assert_series_equal(
            integral, vector_unstructured
        )
