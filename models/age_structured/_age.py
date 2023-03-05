'''Defaults for the age variable common to `model.Model()` and
`_population.Solver()`.'''


# The default maximum age `a_max` for `model.Model()` and
# `_population.model.Model()`.
# This was chosen as the last age where either of the age-dependent
# parameters changes. The age-dependent parameters are `death` and
# `birth.maternity`.
max_default = 12


def check_max(a_max, parameters):
    '''Check that 'a_max' is large enough.'''
    assert a_max >= parameters.death._age_max()
    assert a_max >= parameters.birth._age_max()
