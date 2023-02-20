#!/usr/bin/python3

import time

from context import models
import models
import models.unstructured


if __name__ == '__main__':
    (t_start, t_end, t_step) = (0, 10, 0.001)

    model = models.unstructured.Model(t_step, birth_variation=0)
    solution = model.solve((t_start, t_end))
    equilibrium = model.find_equilibrium(solution.loc[t_end])
    # t0 = time.time()
    # equilibrium_eigvals = model.get_eigenvalues(equilibrium)
    # print('Run time {} sec'.format(time.time() - t0))

    J = model.jacobian(0, equilibrium)
