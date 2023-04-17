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
#       t_step:     1e-2       5e-3       4e-3
# method:
#   sparse_csc  42.8 sec
#   sparse_csr  44.5 sec
#        dense   9.0 sec   87.3 sec  166.6 sec
# dense_memmap  11.5 sec  114.8 sec  382.2 sec
#

import time

import pandas

from context import models


def time_jac(Model, t_step, method, t_solve=10, **kwds):
    model = Model(t_step=t_step,
                  birth_variation=0,
                  _solver_options=dict(_check_matrices=False,
                                       _jacobian_method=method),
                  **kwds)
    t0_eql = time.perf_counter()
    eql = model.find_equilibrium(model.build_initial_conditions(),
                                 t_solve=t_solve)
    t_eql = time.perf_counter() - t0_eql
    print(f'Found equilibrium for {t_step=}, {method=} in {t_eql} sec')
    t0_bld = time.perf_counter()
    model._solver._jacobian
    t_bld = time.perf_counter() - t0_bld
    print(f'Built Jacobian solver for {t_step=}, {method=} in {t_bld} sec')
    t0_jac = time.perf_counter()
    J = model._solver.jacobian(0, eql, eql)
    t_jac = time.perf_counter() - t0_jac
    print(f'Found Jacobian for {t_step=}, {method=} in {t_jac} sec')
    return t_jac


if __name__ == '__main__':
    # Model = models.unstructured.Model
    # Model = models.time_since_entry_structured.Model
    Model = models.age_structured.Model
    kwds = dict(transmission_rate=10)

    t_steps = pandas.Index([4e-3],
                           name='t_step')
    methods = pandas.Index(['dense', 'dense_memmap'],
                           name='method')
    times = pandas.DataFrame(index=t_steps,
                             columns=methods)

    for t_step in t_steps:
        for method in methods:
            times.loc[t_step, method] = time_jac(Model, t_step, method, **kwds)
