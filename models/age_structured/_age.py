'''Defaults for the age variable common to `model.Model()` and
`_population.solver()`.'''


# The default maximum age `a_max` for `age_structured.model.Model()`
# and `age_structured._population._Solver()`.
# This was chosen as the last age where either of the age-dependent
# parameters changes. The age-dependent parameters are `death` and
# `birth.maternity`.
max_default = 12
