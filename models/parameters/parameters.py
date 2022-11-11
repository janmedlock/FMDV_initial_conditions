'''Parameters common to all models.'''

import dataclasses


@dataclasses.dataclass
class Parameters:
    '''Parameters common to all models.'''

    maternal_immunity_waning_rate: float =  1 / 0.37  # per year
    transmission_rate: float = 2.8 * 365              # per year
    progression_rate: float = 1 / 0.5 * 365           # per year
    recovery_rate: float = 1 / 5.7 * 365              # per year
