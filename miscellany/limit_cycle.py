#!/usr/bin/python3
'''Script to find limit cycles.'''

from context import models


T_END = 10


class ModelAge(models.age_structured.Model):
    variables = 'age'


class ModelTSE(models.time_since_entry_structured.Model):
    variables = 'tse'


def filename(model, SAT):
    return f'limit_cycle_{model.variables}_SAT{SAT}.pkl'


def _find_limit_cycle(Model, SAT):
    model = Model(SAT=SAT)
    print(f'{SAT=}: Solving')
    soln = model.solve((0, T_END))
    print(f'{SAT=}: Finding limit cycle')
    lcy = model.find_limit_cycle(model.parameters.period,
                                 T_END % model.parameters.period,
                                 soln.loc[T_END])
    lcy.to_pickle(filename(model, SAT))


def find_limit_cycle_age(SAT):
    return _find_limit_cycle(ModelAge, SAT)


def find_limit_cycle_tse(SAT):
    return _find_limit_cycle(ModelTSE, SAT)


if __name__ == '__main__':
    for SAT in (1, 2, 3):
        find_limit_cycle_age(SAT)
        find_limit_cycle_tse(SAT)
