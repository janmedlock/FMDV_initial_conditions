#!/usr/bin/python3
'''Based on our FMDV work, this is an age-structured model with
periodic birth rate.'''

import models.age_structured


if __name__ == '__main__':
    model_constant = models.age_structured.Model(birth_variation=0)
    print(model_constant.birth_rate.mean)

    model = models.age_structured.Model()
    print(model.birth_rate.mean)
