from enum import Enum
import attr

class AdjacencyType(Enum):
    BINARY = "SHAPE_FILE"
    EXPONENTIAL_DISTANCE = 'EXPONENTIAL_DISTANCE'
    EXPONENTIAL_CENTROID_DISTANCE = 'EXPONENTIAL_CENTROID_DISTANCE'
    COMMON_BORDER_RATIO = 'COMMON_BORDER_RATIO'

    @classmethod
    def from_str(cls, adjacency_type: str) -> 'AdjacencyType':
        try:
            adjacency_type = getattr(cls, adjacency_type.upper())
        except AttributeError:
            raise AttributeError(f"{cls.__class__.__name__} has no {adjacency_type} attribute")
        return 