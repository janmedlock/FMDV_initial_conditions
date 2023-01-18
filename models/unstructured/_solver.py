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
        self._build_matrix_transmission()
        self._build_force_of_infection()
        self._build_scratch()

    def _build_matrices_constant(self):
        C_MM = - (1 / self.model.waning.mean
                  + self.model.death_rate_mean)
        C_SM = 1 / self.model.waning.mean
        C_SS = - self.model.death_rate_mean
        C_EE = - (1 / self.model.progression.mean
                  + self.model.death_rate_mean)
        C_IE = 1 / self.model.progression.mean
        C_II = - (1 / self.model.recovery.mean
                  + self.model.death_rate_mean)
        C_RI = 1 / self.model.recovery.mean
        C_RR = - self.model.death_rate_mean
        C = (self.t_step / 2
             * numpy.array(((C_MM, 0, 0, 0, 0),
                            (C_SM, C_SS, 0, 0, 0),
                            (0, 0, C_EE, 0, 0),
                            (0, 0, C_IE, C_II, 0),
                            (0, 0, 0, C_RI, C_RR))))
        I = numpy.eye(*C.shape)
        self._P = I - C
        self._Q = I + C

    def _build_matrix_birth(self):
        self._Birth = (self.t_step / 2
                       * numpy.array(((0, 0, 0, 0, 1),
                                      (1, 1, 1, 1, 0),
                                      (0, 0, 0, 0, 0),
                                      (0, 0, 0, 0, 0),
                                      (0, 0, 0, 0, 0))))

    def _build_matrix_transmission(self):
        self._Transmission = (self.t_step / 2
                              * numpy.array(((0, 0, 0, 0, 0),
                                             (0, -1, 0, 0, 0),
                                             (0, 1, 0, 0, 0),
                                             (0, 0, 0, 0, 0),
                                             (0, 0, 0, 0, 0))))

    def _build_force_of_infection(self):
        beta = numpy.array((0, 0, 0, 1, 0))
        self._force_of_infection = self.model.transmission.rate * beta

    def _build_scratch(self):
        n = len(self.model.states)
        self._P_new = numpy.empty((n, n))
        self._P_Birth = numpy.empty((n, n))
        self._Q_cur = numpy.empty((n, n))
        self._d_cur = numpy.empty(n)

    def _objective(self, y_new):
        self._P_new[:] = (self._P_Birth
                          - ((self._force_of_infection @ y_new)
                             * self._Transmission))
        return self._P_new @ y_new - self._d_cur

    def _step(self, t_cur, y_cur, y_new):
        '''Do a step.'''
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.birth.rate(t_mid)
        self._P_Birth[:] = (self._P
                            - b_mid * self._Birth)
        self._Q_cur[:] = (self._Q
                          + b_mid * self._Birth
                          + ((self._force_of_infection @ y_cur)
                             * self._Transmission))
        self._d_cur[:] = self._Q_cur @ y_cur
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
        for ell in range(1, len(t)):
            self._step(t[ell - 1], y[ell - 1], y[ell])
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
