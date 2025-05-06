#!/usr/bin/python3
'''Test the unstructured model.'''


import common
from context import models


class TestUnstructuredModel(common.TestModel):
    '''Test the unstructured model.'''
    Model = models.unstructured.Model
