#!/usr/bin/python3

# models.unstructured.Model()
#         base:  0.0 sec
#   sparse_csr:  0.0 sec
#   sparse_csc:  0.0 sec
#        dense:  0.0 sec
#
# models.time_since_entry_structured.Model()
#         base:  3.9 sec
#   sparse_csc:  3.8 sec
#   sparse_csr:  3.6 sec
#        dense: 29.2 sec
#
# models.age_structured.Model(t_step=0.01, transmission_rate=10)
#   sparse_csc: 42.8 sec
#   sparse_csr: 44.5 sec
#        dense:  9.0 sec
# dense_memmap: 18.6 sec
#

import time

from context import models


def time_jac(Model, t_solve=10, **kwds):
    model = Model(birth_variation=0, **kwds)
    eql = model.find_equilibrium(model.build_initial_conditions(),
                                 t_solve=t_solve)
    model._solver._jacobian  # Build matrices outside timer.
    t0 = time.perf_counter()
    J = model._solver.jacobian(0, eql, eql)
    t = time.perf_counter() - t0
    print(f'{t} sec')
    return J


if __name__ == '__main__':
    # Model = models.unstructured.Model
    # Model = models.time_since_entry_structured.Model
    Model = models.age_structured.Model
    kwds = dict(
        t_step=0.01,
        transmission_rate=10,
        _solver_options=dict(
            _check_matrices=False,
            _jacobian_method='dense',
        ),
    )

    J = time_jac(Model, **kwds)
