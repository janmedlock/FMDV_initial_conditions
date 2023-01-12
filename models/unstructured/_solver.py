'''Solver.'''

import numpy
import scipy.optimize

from . import _solution
from .. import _utility


class Solver:
    '''Crank–Nicolson solver.'''

    def __init__(self, model, t_step):
        self.model = model
        self.t_step = t_step
        self._build_matrices_constant()
        self._build_matrix_birth()
        self._build_matrix_infection()
        self._build_force_of_infection()
        self._build_scratch()

    def _build_matrices_constant(self):
        M_MM = - (1 / self.model.waning.mean
                  + self.model.death_rate_mean)
        M_SM = 1 / self.model.waning.mean
        M_SS = - self.model.death_rate_mean
        M_EE = - (1 / self.model.progression.mean
                  + self.model.death_rate_mean)
        M_IE = 1 / self.model.progression.mean
        M_II = - (1 / self.model.recovery.mean
                  + self.model.death_rate_mean)
        M_RI = 1 / self.model.recovery.mean
        M_RR = - self.model.death_rate_mean
        M = numpy.array(((M_MM, 0, 0, 0, 0),
                         (M_SM, M_SS, 0, 0, 0),
                         (0, 0, M_EE, 0, 0),
                         (0, 0, M_IE, M_II, 0),
                         (0, 0, 0, M_RI, M_RR)))
        M_constant = self.t_step / 2 * M
        I = numpy.eye(*M_constant.shape)
        self._A_constant = I - M_constant
        self._B_constant = I + M_constant

    def _build_matrix_birth(self):
        M = numpy.array(((0, 0, 0, 0, 1),
                         (1, 1, 1, 1, 0),
                         (0, 0, 0, 0, 0),
                         (0, 0, 0, 0, 0),
                         (0, 0, 0, 0, 0)))
        self._M_birth = self.t_step / 2 * M

    def _build_matrix_infection(self):
        M = numpy.array(((0, 0, 0, 0, 0),
                         (0, -1, 0, 0, 0),
                         (0, 1, 0, 0, 0),
                         (0, 0, 0, 0, 0),
                         (0, 0, 0, 0, 0)))
        self._M_infection = self.t_step / 2 * M

    def _build_force_of_infection(self):
        beta = numpy.array((0, 0, 0, 1, 0))
        self._force_of_infection = self.model.transmission.rate * beta

    def _build_scratch(self):
        n = len(self.model.states)
        self._A_new = numpy.empty((n, n))
        self._A_constant_birth_new = numpy.empty((n, n))
        self._B_cur = numpy.empty((n, n))
        self._c_cur = numpy.empty(n)

    def _objective(self, y_new):
        self._A_new[:] = (self._A_constant_birth_new
                          - ((self._force_of_infection @ y_new)
                             * self._M_infection))
        return self._A_new @ y_new - self._c_cur

    def _step(self, t_cur, y_cur, y_new):
        '''Do a step.'''
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.birth.rate(t_mid)
        self._A_constant_birth_new[:] = (self._A_constant
                                         - b_mid * self._M_birth)
        self._B_cur[:] = (self._B_constant
                          + b_mid * self._M_birth
                          + ((self._force_of_infection @ y_cur)
                             * self._M_infection))
        self._c_cur[:] = self._B_cur @ y_cur
        result = scipy.optimize.root(self._objective, y_cur)
        assert result.success, f't={t_cur}: {result}'
        y_new[:] = result.x

    def __call__(self, t_span, y_0,
                 t=None, y=None, _solution_wrap=True):
        '''Solve. `y` is storage for the solution, which will be built if not
        provided. `_solution_wrap=False` skips wrapping the solution in
        `_solution.Solution()` for speed.'''
        if t is None:
            t = _utility.build_t(*t_span, self.t_step)
        if y is None:
            y = numpy.empty((len(t), *numpy.shape(y_0)))
        y[0] = y_0
        for k in range(1, len(t)):
            self._step(t[k - 1], y[k - 1], y[k])
        if _solution_wrap:
            return _solution.Solution(y, t, states=self.model.states)
        else:
            return y


def solve(model, t_span, t_step, y_0,
          t=None, y=None, _solution_wrap=True):
    '''Solve the ODE defined by the derivatives in `model`.'''
    solver = Solver(model, t_step)
    return solver(t_span, y_0,
                  t=t, y=y, _solution_wrap=_solution_wrap)
