#!/usr/bin/python3

from pandas.testing import assert_series_equal

from context import models


hows = ('survival', 'all_in_first')


if __name__ == '__main__':
    # Keep vectors short for fast testing.
    t_step = 1e-1

    model_u = models.unstructured.Model(t_step=t_step)
    model_a = models.age_structured.Model(t_step=t_step)
    model_z = models.time_since_entry_structured.Model(t_step=t_step)
    model_c = models.combination.Model(t_step=t_step)

    y_u = model_u.build_initial_conditions()
    y_a = model_a.build_initial_conditions()
    assert_series_equal(model_a.integral_over_a(y_a), y_u)
    for how in hows:
        y_z = model_z.build_initial_conditions(how=how)
        y_c = model_c.build_initial_conditions(how=how)
        assert_series_equal(model_z.integral_over_z(y_z), y_u)
        assert_series_equal(model_c.integral_over_a_and_z(y_c), y_u)
