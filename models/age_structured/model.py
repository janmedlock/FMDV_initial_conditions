'''Based on our FMDV work, this is an age-structured model.'''

from . import _parameters
from .. import _model


class _Model(_model.Model):
    '''Base class for age-structured models.'''


class ModelBirthConstant(_Model):
    '''Age-structured model with constant birth rate.'''

    _Parameters = _parameters.ParametersBirthConstant


class ModelBirthPeriodic(_Model):
    '''Age-structured model with periodic birth rate.'''

    _Parameters = _parameters.ParametersBirthPeriodic
