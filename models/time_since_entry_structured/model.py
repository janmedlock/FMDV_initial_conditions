'''Based on our FMDV work, this is a time-since-entry-structured model.'''

import numpy
import pandas

from . import _solver
from .. import _model
from .. import _equilibrium
from .. import _utility


class Model(_model.AgeIndependent):
    '''Time-since-entry-structured model.'''

    states_with_z = ['maternal_immunity', 'exposed', 'infectious']

    def __init__(self, z_step=0.001, z_max=3, **kwds):
        super().__init__(**kwds)
        self.z_step = z_step
        self.z = _utility.build_t(0, z_max, self.z_step)
        self._solver = _solver.Solver(self)

    def _index_states_z(self):
        # Build a `pandas.DataFrame()` with columns 'state' and
        # 'time_since_entry' to be converted into a `pandas.MultiIndex()`.
        zvals = lambda state: (self.z
                               if state in self.states_with_z
                               else [numpy.NaN])
        dfr = pandas.concat(
            pandas.DataFrame({'state': state,
                              'time_since_entry': zvals(state)})
            for state in self.states
        )
        # Make 'state' categorical and ordered.
        dtype = {'state': pandas.CategoricalDtype(self.states, ordered=True)}
        return pandas.MultiIndex.from_frame(dfr.astype(dtype))

    def Solution(self, y, t=None):
        '''A solution.'''
        states_z = self._index_states_z()
        if t is None:
            return pandas.Series(y, index=states_z)
        else:
            t = pandas.Index(t, name='time')
            return pandas.DataFrame(y, index=t, columns=states_z)

    def build_initial_conditions(self):
        '''Build the initial conditions.'''
        K = len(self.z)
        # Totals over time since entry.
        M_bar = 0
        E_bar = 0
        I_bar = 0.01
        R = 0
        S = 1 - M_bar - E_bar - I_bar - R
        # All in the first time since entry.
        n = numpy.hstack([1 / self.z_step, numpy.zeros(K - 1)])
        (m, e, i) = numpy.outer((M_bar, E_bar, I_bar), n)
        return numpy.hstack((m, S, e, i, R))

    def _survival_scaled(self, waiting_time):
        survival = waiting_time.survival(self.z)
        # Scale to integrate to 1.
        total = survival.sum() * self.z_step
        return survival / total

    def initial_conditions_from_unstructured(self, Y):
        '''Build initial conditions from the unstructured `Y`.'''
        (M, S, E, I, R) = Y
        m = M * self._survival_scaled(self.waning)
        e = E * self._survival_scaled(self.progression)
        i = I * self._survival_scaled(self.recovery)
        return numpy.hstack((m, S, e, i, R))

    def solve(self, t_span,
              y_start=None, t=None, y=None, display=False,
              _solution_wrap=True):
        '''Solve the ODEs.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        soln = self._solver.solve(t_span, y_start,
                                  t=t, y=y, display=display,
                                  _solution_wrap=_solution_wrap)
        _utility.assert_nonnegative(soln)
        return soln

    def _get_weights(self):
        K = len(self.z)
        z_steps = self.z_step * numpy.ones(K)
        w_state = [z_steps if state in self.states_with_z else 1
                   for state in self.states]
        return numpy.hstack(w_state)

    def find_equilibrium(self, eql_guess, t=0, **root_kwds):
        '''Find an equilibrium of the model.'''
        if not 'method' in root_kwds:
            root_kwds['method'] = 'krylov'
        weights = self._get_weights()
        eql = _equilibrium.find(self, eql_guess, t,
                                weights=weights, **root_kwds)
        _utility.assert_nonnegative(eql * weights)
        return eql
