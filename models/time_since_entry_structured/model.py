'''Based on our FMDV work, this is a time-since-entry-structured model.'''

import numpy
import pandas

from . import _solver
from .. import model
from .. import _utility


class Model(model.AgeIndependent):
    '''Time-since-entry-structured model.'''

    states_with_z = ['maternal_immunity', 'exposed', 'infectious']

    def __init__(self, z_step=0.1, z_max=50, **kwds):
        super().__init__(**kwds)
        self.z_step = z_step
        self.z = _utility.build_t(0, z_max, self.z_step)

    def Solution(self, y, t=None):
        '''A solution.'''
        tuples = []
        for state in self.states:
            if state in self.states_with_z:
                tuples.extend((state, z) for z in self.z)
            else:
                tuples.append((state, None))
        states_z = pandas.MultiIndex.from_tuples(tuples,
                                                 names=['state',
                                                        'time_since_entry'])
        if t is None:
            return pandas.Series(y, index=states_z)
        else:
            t = pandas.Index(t, name='time')
            return pandas.DataFrame(y, index=t, columns=states_z)

    def __call__(self, t, y):
        raise NotImplementedError

    def jacobian(self, t, y):
        raise NotImplementedError

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

    def solver(self):
        '''Only initialize the solver once.'''
        return _solver.Solver(self)

    def solve(self, t_span,
              y_start=None, t=None, y=None, _solution_wrap=True):
        '''Solve the ODEs.'''
        if y_start is None:
            y_start = self.build_initial_conditions()
        solver = self.solver()
        soln = solver(t_span, y_start,
                      t=t, y=y, _solution_wrap=_solution_wrap)
        _utility.assert_nonnegative(soln)
        return soln
