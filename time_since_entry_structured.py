#!/usr/bin/python3
'''Based on our FMDV work, this is a time-since-entry-structured model
with periodic birth rate.'''

import time

import models.time_since_entry_structured


if __name__ == '__main__':
    (t_start, t_end) = (0, 1)

    model_constant = models.time_since_entry_structured.Model(
        birth_variation=0)
    t0 = time.time()
    solution_constant = model_constant.solve((t_start, t_end))
    print('Runtime {} sec.'.format(time.time() - t0))

    # model = models.time_since_entry_structured.Model()
