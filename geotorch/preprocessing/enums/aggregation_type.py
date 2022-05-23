from enum import Enum
import attr

class AggregationType(Enum):
    COUNT = "COUNT"
    SUM = 'SUM'
    AVG = 'AVG'
    MIN = 'MIN'
    MAX = 'MAX'

    def __str__(self):
        return str(self.value)

    @classmethod
    def from_str(cls, aggregation_type: str) -> 'AggregationType':
        try:
            aggregation_type = getattr(cls, aggregation_type.upper())
        except AttributeError:
            raise AttributeError(f"{cls.__class__.__name__} has no {aggregation_type} attribute")
        return 