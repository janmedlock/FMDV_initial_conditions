#!/usr/bin/python3
'''Time model methods vs. step size.

Model = models.age_structured.Model
SAT = 1
transmission_rate_factor = 0.5
progression_mean_factor = t_step / 1e-3
+--------+-----------------------------------------------------------------+
|        |                  jacobian_method                                |
| t_step +-----------+--------------+------------+------------+------------+
|        |   dense   | dense_memmap | sparse_csr | sparse_csc | sparse_bsr |
+--------+-----------+--------------+------------+------------+------------+
|  0.010 |    11 sec |      164 sec |   3115 sec |   2271 sec |   3543 sec |
|  0.005 |   413 sec |     1069 sec |  21957 sec |  21298 sec |    OOM     |
|  0.003 |   143 sec |              |            |            |            |
|  0.002 | 20725 sec |              |            |            |            |
|  0.001 |           |              |            |            |            |
+--------+-----------+--------------+------------+------------+------------+
'''

import numpy

from context import models
import time_model


def build_model(Model, t_step, jacobian_method=None,
                transmission_rate_factor=1, **parameters_kws):
    # Attach timers to `Model`.
    time_model.time_model(Model)
    # Build parameters.
    parameters = models.parameters.Parameters(**parameters_kws)
    if transmission_rate_factor != 1:
        # Scale 'transmission_rate' by `transmission_rate_factor`.
        parameters.transmission_rate *= transmission_rate_factor
    # Scale 'progression_mean' by `t_step / 1e-3`, i.e.  1 for the
    # default `t_step` of 1e-3 and slower for larger `t_step`.
    progression_mean_scale = t_step / 1e-3
    parameters.progression_mean *= progression_mean_scale
    solver_kwds = {}
    if jacobian_method is not None:
        solver_kwds['_jacobian_method'] = jacobian_method
    return Model(t_step=t_step,
                 solver_kwds=solver_kwds,
                 **parameters)


if __name__ == '__main__':
    Model = models.age_structured.Model
    T_END = 10

    model = build_model(
        Model=Model,
        t_step=10e-3,
        jacobian_method='dense',
        SAT=1,
        transmission_rate_factor=0.3,
        birth_variation=0,
    )

    soln = model.solve((0, T_END))
    eql = model.find_equilibrium(soln.iloc[-1])
    k = 2 * len(model.a)
    exps_eql = model.get_eigenvalues(eql, k=k, verbose=True)
    mults_eql = numpy.exp(exps_eql)
    print(f'{mults_eql=}')
