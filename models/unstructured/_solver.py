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
        H = numpy.identity(5)
        # Rates.
        death = self.model.death_rate_mean
        waning = 1 / self.model.waning.mean
        progression = 1 / self.model.progression.mean
        recovery = 1 / self.model.recovery.mean
        F = numpy.array((
            (- (waning + death), 0, 0, 0, 0),
            (waning, - death, 0, 0, 0),
            (0, 0, - (progression + death), 0, 0),
            (0, 0, progression, - (recovery + death), 0),
            (0, 0, 0, recovery, - death)
        ))
        self._HF0 = H - self.t_step / 2 * F
        self._HF1 = H + self.t_step / 2 * F

    def _build_matrix_birth(self):
        self._B = (self.t_step / 2
                   * numpy.array(((0, 0, 0, 0, 1),
                                  (1, 1, 1, 1, 0),
                                  (0, 0, 0, 0, 0),
                                  (0, 0, 0, 0, 0),
                                  (0, 0, 0, 0, 0))))

    def _build_matrix_transmission(self):
        self._T = (self.t_step / 2
                   * numpy.array(((0, 0, 0, 0, 0),
                                  (0, -1, 0, 0, 0),
                                  (0, 1, 0, 0, 0),
                                  (0, 0, 0, 0, 0),
                                  (0, 0, 0, 0, 0))))

    def _build_force_of_infection(self):
        self._beta = (self.model.transmission.rate
                      * numpy.array((0, 0, 0, 1, 0)))

    def _build_scratch(self):
        n = len(self.model.states)
        self._HFB0 = numpy.empty((n, n))
        self._HFBT0 = numpy.empty((n, n))
        self._HFBT1 = numpy.empty((n, n))
        self._HFBTx1 = numpy.empty(n)

    def _objective(self, y_new):
        self._HFBT0[:] = (self._HFB0
                          - (self._beta @ y_new) * self._T)
        return self._HFBT0 @ y_new - self._HFBTx1

    def _step(self, t_cur, y_cur, y_new):
        '''Do a step.'''
        t_mid = t_cur + 0.5 * self.t_step
        b_mid = self.model.birth.rate(t_mid)
        self._HFB0[:] = (self._HF0
                         - b_mid * self._B)
        self._HFBT1[:] = (self._HF1
                          + b_mid * self._B
                          + (self._beta @ y_cur) * self._T)
        self._HFBTx1[:] = self._HFBT1 @ y_cur
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
