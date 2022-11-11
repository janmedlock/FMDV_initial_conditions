#!/usr/bin/python3
'''Based on our FMDV work, this is an age-structured model with
periodic birth rate.'''

import models.age_structured


if __name__ == '__main__':
    model_constant = models.age_structured.ModelBirthConstant()

    model_periodic = models.age_structured.ModelBirthPeriodic()
