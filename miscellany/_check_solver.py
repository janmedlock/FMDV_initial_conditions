'''Utilities for checking the model solver matrices.'''

from context import models
from models._utility import sparse


class Base:
    def __init__(self, model):
        self.model = model

    def check_matrices(self):
        solver = self.model._solver
        assert sparse.equals(self.beta(), solver.beta)
        assert sparse.equals(self.Hq('new'), solver.H_new)
        assert sparse.equals(self.Hq('cur'), solver.H_cur)
        assert sparse.equals(self.Fq('new'), solver.F_new)
        assert sparse.equals(self.Fq('cur'), solver.F_cur)
        assert sparse.equals(self.Tq('new'), solver.T_new)
        assert sparse.equals(self.Tq('cur'), solver.T_cur)
        assert sparse.equals(self.B(), solver.B)
