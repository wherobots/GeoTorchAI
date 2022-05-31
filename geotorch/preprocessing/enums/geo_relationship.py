from enum import Enum
import attr

class GeoRelationship(Enum):
    CONTAINS = "ST_Contains" #
    INTERSECTS = 'ST_Intersects'
    TOUCHES = 'ST_Touches'
    WITHIN = 'ST_Within'

    def __str__(self):
        return str(self.value)

    @classmethod
    def from_str(cls, geo_relationship: str) -> 'GeoRelationship':
        try:
            geo_relationship = getattr(cls, geo_relationship.upper())
        except AttributeError:
            raise AttributeError(f"{cls.__class__.__name__} has no {geo_relationship} attribute")
        return 
