#!/usr/bin/python3

import matplotlib.pyplot
import numpy

from context import models
import models.age_structured
import models.time_since_entry_structured
import models.unstructured


if __name__ == '__main__':
    (t_start, t_end) = (0, 10)
    params = dict(birth_variation=0)

    model_us = models.unstructured.Model(**params)
    solution_us = model_us.solve((t_start, t_end))
    equilibrium_us = model_us.find_equilibrium(solution_us.loc[t_end])

    model_tses = models.time_since_entry_structured.Model(**params)
    y_start_tses = model_tses.initial_conditions_from_unstructured(
        equilibrium_us)
    solution_tses = model_tses.solve((t_start, t_end),
                                     y_start=y_start_tses)
    totals_tses = solution_tses.time_since_entry.aggregate()
    totals_tses.plotting.solution()

    model_as = models.age_structured.Model(**params)
    y_start_as = model_as.initial_conditions_from_unstructured(
        equilibrium_us)
    solution_as = model_as.solve((t_start, t_end),
                                 y_start=y_start_as)
    totals_as = solution_as.age.aggregate()
    totals_as.plotting.solution()

    matplotlib.pyplot.show()
