'''Recovery.'''

import dataclasses


@dataclasses.dataclass
class Recovery:
    '''Recovery.'''

    recovery_mean: float = 5.7 / 365  # years
    recovery_shape: float = 11.8      # unitless
