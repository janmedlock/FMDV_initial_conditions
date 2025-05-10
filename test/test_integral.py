#!/usr/bin/python3
'''Test integrals.'''

import inspect
import operator

import pandas.testing
import pytest

from context import models


def _model_class(model_name):
    '''Get the model class.'''
    return getattr(models, model_name).Model


# The keywords used when initiating each model.
MODEL_KWS = {
    # Keep vectors short for fast testing ...
    't_step': 1e-2,
    # ... which requires turning off matrix checks.
    'solver_kwds': {
        '_check_matrices': False,
    }
}


def _model(model_name):
    '''Build a model instance.'''
    return _model_class(model_name)(**MODEL_KWS)


def _state_method(model):
    '''The method used to build a state vector for `model`.'''
    return model.build_initial_conditions


def _state(model, **kws):
    '''A state vector for `model`.'''
    return _state_method(model)(**kws)


# For each structured model, the integral method that integrates over
# the structure variables.
_INTEGRALS = {
    'time_since_entry_structured': operator.attrgetter('integral_over_z'),
    'age_structured': operator.attrgetter('integral_over_a'),
    'combination': operator.attrgetter('integral_over_a_and_z'),
}


def _integral_method(model):
    '''The integral method for `model`.'''
    for (model_name, integral_attr) in _INTEGRALS.items():
        # Use `type()` to get exact matches, not subclasses.
        if type(model) is _model_class(model_name):
            return integral_attr(model)
    raise ValueError(f'Unknown {type(model)=}!')


_MODELS_STRUCTURED = _INTEGRALS.keys()


# For models whose `_state_method(model)` takes a `how` argument, the
# allowed values of the `how` argument.
_HOW_VALUES = (
    'survival',
    'all_in_first',
)


def _state_method_takes_how(model_name):
    '''Whether `_state_method(model)` has the `how` argument.'''
    model_class = _model_class(model_name)
    method = _state_method(model_class)
    sig = inspect.signature(method)
    return 'how' in sig.parameters


def _how_values(model_name):
    if _state_method_takes_how(model_name):
        return _HOW_VALUES
    return (None, )


_model_name_and_how_values = [
    (model_name, how)
    for model_name in _MODELS_STRUCTURED
    for how in _how_values(model_name)
]


@pytest.fixture(scope='module')
def state_unstructured():
    '''A state vector for the unstructured model.'''
    model = _model('unstructured')
    return _state(model)


@pytest.fixture(scope='class')
def model(model_name):
    '''A model instance.'''
    return _model(model_name)


@pytest.fixture
def state_integrated(model, how):
    '''The integral of the state vector.'''
    kws = {'how': how} if how is not None else {}
    state = _state(model, **kws)
    return _integral_method(model)(state)


@pytest.mark.parametrize('model_name, how',
                         _model_name_and_how_values,
                         scope='class')
class TestIntegral:
    '''Test integrals.'''
    def test_integral(self, state_integrated, state_unstructured):
        pandas.testing.assert_series_equal(
            state_integrated, state_unstructured
        )
