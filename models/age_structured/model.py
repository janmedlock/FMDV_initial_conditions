'''Based on our FMDV work, this is an age-structured model.'''

from . import parameters
from .. import model


class _Model(model.Model):
    '''Base class for age-structured models.'''


class ModelBirthPeriodic(_Model):
    '''Age-structured model with periodic birth rate.'''

    _Parameters = parameters.ParametersBirthPeriodic


class ModelBirthConstant(_Model):
    '''Age-structured model with constant birth rate.'''

    _Parameters = parameters.ParametersBirthConstant
