'''Progression.'''

import dataclasses


@dataclasses.dataclass
class Progression:
    '''Progression.'''

    progression_mean: float = 0.5 / 365  # years
    progression_shape: float = 1.2       # unitless
