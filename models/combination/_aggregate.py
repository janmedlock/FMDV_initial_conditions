'''`pandas` accessor to integrate over age and time since entry.'''

from ..age_structured._aggregate import AgeAccessor
from ..time_since_entry_structured._aggregate import TimeSinceEntryAccessor
from .._model import aggregate


class CombinationAccessor(aggregate.BaseAccessor):
    '''API to integrate over age and time since entry.'''

    _accessor_name = 'age_and_time_since_entry'

    @classmethod
    def _aggregate(cls, obj):
        '''Integrate over age and time since entry.'''
        return AgeAccessor._aggregate(
            TimeSinceEntryAccessor._aggregate(obj)
        )


CombinationAccessor._register()
