#!/usr/bin/python3

from pandas.testing import assert_series_equal

from context import models


# Keep vectors short for fast testing.
t_step = 1e-1


model_u = models.unstructured.Model()
y_u = model_u.build_initial_conditions()

model_a = models.age_structured.Model(t_step=t_step)
y_a = model_a.build_initial_conditions()
assert_series_equal(model_a.integral_over_a(y_a), y_u)

model_z = models.time_since_entry_structured.Model(t_step=t_step)
y_z = model_z.build_initial_conditions()
assert_series_equal(model_z.integral_over_z(y_z), y_u)

model_c = models.combination.Model(t_step=t_step)
y_c = model_c.build_initial_conditions()
assert_series_equal(model_c.integral_over_a_and_z(y_c), y_u)
assert_series_equal(model_c.integral_over_z(y_c), y_a)
assert_series_equal(model_c.integral_over_a(y_c), y_z)
