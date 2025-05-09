#!/usr/bin/python3
'''Test the unstructured model.'''


import model
from context import models


class TestModel(model.TestModel):
    '''Test the unstructured model.'''
    Model = models.unstructured.Model
