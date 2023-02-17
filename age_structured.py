#!/usr/bin/python3
'''Based on our FMDV work, this is an age-structured model with
periodic birth rate.'''

import time

import matplotlib.pyplot

import models
import models.age_structured


if __name__ == '__main__':
    (t_start, t_end) = (0, 1)

    # model_constant = models.age_structured.Model(birth_variation=0)
    # solution_constant = model_constant.solve((t_start, t_end))
    # totals_constant = solution_constant.age.aggregate()
    # ax_solution = totals.solution.plot_solution()

    model = models.age_structured.Model()
    t0 = time.time()
    solution = model.solve((t_start, t_end))
    print('Runtime {} sec.'.format(time.time() - t0))
    totals = solution.age.aggregate()
    # totals.solution.plot_solution(ax=ax_solution, legend=False)
    totals.solution.plot_solution()

    matplotlib.pyplot.show()
