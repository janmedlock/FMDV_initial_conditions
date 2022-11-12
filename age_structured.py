#!/usr/bin/python3
'''Based on our FMDV work, this is an age-structured model with
periodic birth rate.'''

import models.age_structured


if __name__ == '__main__':
    model_constant = models.age_structured.ModelBirthConstant()
    print(model_constant.parameters.birth_rate_mean)

    model_periodic = models.age_structured.ModelBirthPeriodic()
    print(model_periodic.parameters.birth_rate_mean)
