from enum import Enum
import attr

class GeoFileType(Enum):
    SHAPE_FILE = "SHAPE_FILE"
    JSON_FILE = "JSON_FILE"
    WKB_FILE = "WKB_FILE"
    WKT_FILE = "WKT_FILE"

    @classmethod
    def from_str(cls, file_type: str) -> 'GeoFileType':
        try:
            file_type = getattr(cls, file_type.upper())
        except AttributeError:
            raise AttributeError(f"{cls.__class__.__name__} has no {file_type} attribute")
        return 