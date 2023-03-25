'''Base for age-structured models.'''

import abc


class Model(metaclass=abc.ABCMeta):
    '''Base for age-structured models.'''

    @property
    @abc.abstractmethod
    def _Solver(self):
        '''The solver class.'''

    # The default maximum age `a_max` for `model.Model()` and
    # `_population.model.Model()`.
    # This was chosen as the last age where either of the age-dependent
    # parameters changes. The age-dependent parameters are `death` and
    # `birth.maternity`.
    _a_max_default = 12

    def __init__(self, a_max=_a_max_default, **kwds):
        assert a_max > 0
        self.a_max = a_max
        super().__init__(**kwds)

    def _check_a_max(self):
        '''Check that `a_max` is large enough.'''
        assert self.a_max >= self.parameters.birth._age_max()
        assert self.a_max >= self.parameters.death._age_max()

    def _init_post(self):
        '''Final initialization.'''
        self._check_a_max()
        assert self.a_step > 0
        super()._init_post()

    @property
    def a_step(self):
        return self._Solver._get_a_step(self.t_step)
