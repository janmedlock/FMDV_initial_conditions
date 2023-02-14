'''Context for the current directory.'''

import pathlib
import sys

_path = pathlib.Path(__file__).parents[1]
sys.path.append(str(_path))
import models
sys.path.pop()
