from .euro_sat import EuroSATDataset
from .sat6 import SAT6Dataset
from .sat4 import SAT4Dataset
from .slum_detection import SlumDetectionDataset
from .segmentation import Cloud38Dataset
from .processed_dataset import ProcessedDataset
from .processed_dataset_extra_features import ProcessedDatasetWithExtraFeatures

__all__ = ["EuroSATDataset", "SAT6Dataset", "SAT4Dataset", "SlumDetectionDataset", "Cloud38Dataset", "ProcessedDataset", "ProcessedDatasetWithExtraFeatures"]
