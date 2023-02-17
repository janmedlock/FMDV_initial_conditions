#!/usr/bin/python3
'''Based on our FMDV work, this is an age-structured model with
periodic birth rate.'''

import models.age_structured


if __name__ == '__main__':
    (t_start, t_end) = (0, 1)

    model_constant = models.age_structured.Model(birth_variation=0)
    solution_constant = model_constant.solve((t_start, t_end))

    # model = models.age_structured.Model(**kwds)
