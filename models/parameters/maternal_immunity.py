'''Maternal immunity waning.'''

import dataclasses


@dataclasses.dataclass
class Waning:
    '''Maternal immunity waning.'''

    maternal_immunity_duration_mean: float = 0.37   # years
    maternal_immunity_duration_shape: float = 1.19  # unitless
