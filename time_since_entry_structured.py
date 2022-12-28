#!/usr/bin/python3
'''Based on our FMDV work, this is a time-since-entry-structured model
with periodic birth rate.'''

import models.time_since_entry_structured


if __name__ == '__main__':
    model_constant = models.time_since_entry_structured.Model(
        birth_variation=0)
    print(model_constant.birth.mean)

    model = models.time_since_entry_structured.Model()
    print(model.birth.mean)
