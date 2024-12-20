#!/usr/bin/python3
'''Test the model Jacobian.'''

import matplotlib.pyplot
import numpy
import scipy.linalg

from context import models


if __name__ == '__main__':
    model = models.unstructured.Model(birth_variation=0)
    T_END = 10

    soln = model.solve((0, T_END))
    eql = model.find_equilibrium(soln.iloc[-1])

    mu = model.parameters.death_rate_mean
    omega = 1 / model.parameters.waning.mean
    rho = 1 / model.parameters.progression.mean
    gamma = 1 / model.parameters.recovery.mean
    beta = model.parameters.transmission.rate
    b = model.parameters.birth.mean
    (m, s, e, i, r) = eql
    jac = numpy.array([
        [- omega - mu, 0, 0, 0, b],
        [b + omega, b - beta * i - mu, b, b - beta * s, 0],
        [0, beta * i, - rho - mu, beta * s, 0],
        [0, 0, rho, - gamma - mu, 0],
        [0, 0, 0, gamma, - mu]
    ])

    t_step = model.t_step
    I = numpy.eye(len(eql))
    # J = F + beta x T + b B
    # M_new = I - t_step / 2 * J
    # M_cur = I + t_step / 2 * J
    M = {
        q: model.solver._jacobian._M(q, eql, b)
        for q in ('cur', 'new')
    }
    assert numpy.allclose(I - M['new'],
                          M['cur'] - I)
    J = (I - M['new']) / (t_step / 2)
    assert numpy.allclose(J, jac)
    # M_new @ D = M_cur
    D = scipy.linalg.solve(M['new'], M['cur'])
    # J \approx (D - I) / t_step
    J_approx = (D - I) / t_step

    (fig, axes) = matplotlib.pyplot.subplots(1, 2)
    CMAP = 'PiYG'
    vmax = max(numpy.abs(jac).max(),
               numpy.abs(J_approx).max())
    axes[0].matshow(jac, cmap=CMAP, vmax=vmax, vmin=-vmax)
    axes[1].matshow(J_approx, cmap=CMAP, vmax=vmax, vmin=-vmax)
    matplotlib.pyplot.show()
