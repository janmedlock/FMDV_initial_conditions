'''Based on our FMDV work, this is a time-since-entry-structured model.'''

import numpy
import pandas

from . import _solver
from .. import _model
from .. import _utility


class Model(_model.AgeIndependent):
    '''Time-since-entry-structured model.'''

    _Solver = _solver.Solver

    _root_kwds = dict(method='krylov')

    states_with_z = ['maternal_immunity', 'exposed', 'infectious']

    def __init__(self, z_step=0.001, z_max=3, **kwds):
        self.z_step = z_step
        self.z = _utility.build_t(0, z_max, self.z_step)
        super().__init__(**kwds)

    def _build_solution_index(self, states):
        '''Build the solution index.'''
        # Build a `pandas.DataFrame()` with columns 'state' and
        # 'time_since_entry' to be converted into a `pandas.MultiIndex()`.
        zvals = lambda state: (self.z
                               if state in self.states_with_z
                               else [numpy.NaN])
        dfr = pandas.concat(
            pandas.DataFrame({'state': state,
                              'time_since_entry': zvals(state)})
            for state in states
        )
        # Make 'state' categorical and ordered.
        dfr = dfr.astype({'state': states.dtype})
        states_z = pandas.MultiIndex.from_frame(dfr)
        return states_z

    def _build_weights(self):
        '''Build weights for the state vector.'''
        K = len(self.z)
        z_steps = self.z_step * numpy.ones(K)
        w_state = [z_steps if state in self.states_with_z else 1
                   for state in self.states]
        return numpy.hstack(w_state)

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
