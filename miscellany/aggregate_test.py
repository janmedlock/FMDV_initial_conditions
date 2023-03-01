#!/usr/bin/python3

import pandas

from context import models


step = 0.001
a_max = 3 * step
z_max = 2 * step


model_u = models.unstructured.Model()
y_u = model_u.build_initial_conditions()

model_a = models.age_structured.Model(a_step=step, a_max=a_max)
y_a = model_a.build_initial_conditions()
pandas.testing.assert_series_equal(y_a.age.aggregate(), y_u)

model_z = models.time_since_entry_structured.Model(z_step=step, z_max=z_max)
y_z = model_z.build_initial_conditions()
pandas.testing.assert_series_equal(y_z.time_since_entry.aggregate(), y_u)

model_c = models.combination.Model(step=step, a_max=a_max, z_max=z_max)
y_c = model_c.build_initial_conditions()
pandas.testing.assert_series_equal(y_c.age_and_time_since_entry.aggregate(),
                                   y_u)
pandas.testing.assert_series_equal(y_c.time_since_entry.aggregate(), y_a)
pandas.testing.assert_series_equal(y_c.age.aggregate(), y_z)
