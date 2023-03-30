#!/usr/bin/python3

import numpy
import scipy.sparse

from context import models
import models._utility

import solver_test


class Tester(solver_test.Tester):
    '''Test the age– and time-since-entry–structured solver.'''

    def beta(self):
        pass

    def H(self, q):
        pass

    def F(self, q):
        pass

    def T(self, q):
        pass

    def B(self):
        pass


if __name__ == '__main__':
    model = models.combination.Model()
    tester = Tester(model)
    tester.test()
