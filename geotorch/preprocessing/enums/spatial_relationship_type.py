from enum import Enum
import attr

class SpatialRelationshipType(Enum):
    CONTAINS = "ST_Contains" #
    INTERSECTS = 'ST_Intersects'
    TOUCHES = 'ST_Touches'
    WITHIN = 'ST_Within'

    def __str__(self):
        return str(self.value)

    @classmethod
    def from_str(cls, spatial_relationship_type: str) -> 'SpatialRelationshipType':
        try:
            spatial_relationship_type = getattr(cls, spatial_relationship_type.upper())
        except AttributeError:
            raise AttributeError(f"{cls.__class__.__name__} has no {spatial_relationship_type} attribute")
        return 