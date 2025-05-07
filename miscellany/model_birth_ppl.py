#!/usr/bin/python3
'''Test the periodic piecewise-linear birth rate.'''

import sys

from context import models
sys.path.append('..')
import model_unstructured
sys.path.pop()


# Monkeypatch to use piecewise-linear birth rate.
models.birth.BirthPeriodic = models.birth.BirthPeriodicPiecewiseLinear

if __name__ == '__main__':
    model_unstructured.run(
        model_unstructured.SATS,
        model_unstructured.BIRTH_CONSTANTS,
    )
