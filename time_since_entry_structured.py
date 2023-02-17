#!/usr/bin/python3
'''Based on our FMDV work, this is a time-since-entry-structured model
with periodic birth rate.'''

import matplotlib.pyplot

import models
import models.time_since_entry_structured


if __name__ == '__main__':
    (t_start, t_end) = (0, 10)

    model_constant = models.time_since_entry_structured.Model(
        birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end))
    totals_constant = solution_constant.time_since_entry.aggregate()
    ax_solution = totals_constant.solution.plot_solution()

    model = models.time_since_entry_structured.Model()
    solution = model.solve((t_start, t_end))
    totals = solution.time_since_entry.aggregate()
    totals.solution.plot_solution(ax=ax_solution, legend=False)

    matplotlib.pyplot.show()
