
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from geotorchai.utility.exceptions import InvalidParametersException
from geotorchai.utility._download_utils import _download_cdsapi_files
import xarray as xr


class Temperature(Dataset):
    '''
    This dataset is based on https://github.com/jwwthu/DL4Traffic/tree/main/TaxiBJ21

    Parameters
    ..........
    root (String) - Path to the dataset if it is already downloaded. If not downloaded, it will be downloaded in the given path.
    history_length (Int) - Length of history data in sequence of each sample
    prediction_length (Int) - Length of prediction data in sequence of each sample
    download (Boolean, Optional) - Set to True if dataset is not available in the given directory. Default: False
    years (List, Optional) - Dataset will be downloaded for the given years
    '''

    ALL_YEARS = [
    '1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993',
    '1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008',
    '2009','2010','2011','2012','2013','2014','2015','2016','2017','2018'
    ]


    def __init__(self, root, history_length, prediction_length, download=False, years=ALL_YEARS):
        super().__init__()

        self.all_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
        self.all_days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
        '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
        self.all_times = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00',
        '11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']

        self.variable = 'temperature'
        self.level_type = 'pressure'
        self.pressure_level = '850'
        self.grid = [5.625,2.8125]
        self.product_type = 'reanalysis'
        self.format_name = 'netcdf'

        if download:
            _download_cdsapi_files(root, self.variable, years, self.all_months, self.all_days, self.all_times, self.level_type, pressure_level = self.pressure_level, grid = self.grid, product_type = self.product_type, format_name = self.format_name)

        data_dir = self._get_path(root)
        arr = xr.open_mfdataset(f'{data_dir}/*.nc', combine='by_coords')
        self.full_data = arr['t'].values

        self.timesteps = self.full_data.shape[0]
        self.grid_height = self.full_data.shape[1]
        self.grid_width = self.full_data.shape[2]

        self.full_data = self.full_data.reshape((self.timesteps, 1, self.grid_height, self.grid_width))

        self._generate_sequence_data(history_length, prediction_length)



    ## This method returns the total number of timesteps in the generated dataset
    def get_timesteps(self):
        return self.timesteps



    ## This method returns the height of the grid in the generated dataset
    def get_grid_height(self):
        return self.grid_height



    ## This method returns the width of the grid in the generated dataset
    def get_grid_width(self):
        return self.grid_width



    def _generate_sequence_data(self, history_length, prediction_length):
        self.X_data = []
        self.Y_data = []
        total_length = self.full_data.shape[0]
        for end_idx in range(history_length + prediction_length, total_length):
            predict_frames = self.full_data[end_idx-prediction_length:end_idx]
            history_frames = self.full_data[end_idx-prediction_length-history_length:end_idx-prediction_length]
            self.X_data.append(history_frames)
            self.Y_data.append(predict_frames)
        self.X_data = np.stack(self.X_data)
        self.Y_data = np.stack(self.Y_data)
        


    def __len__(self) -> int:
        return len(self.Y_data)


    def __getitem__(self, index: int):
        sample = {"x_data": self.X_data[index], "y_data": self.Y_data[index]}
        return sample


    def _get_path(self, root_dir):
        queue = [root_dir]
        while queue:
            data_dir = queue.pop(0)
            folders = os.listdir(data_dir)
            for folder in folders:
                if folder.endswith(".nc"):
                    return data_dir

            for folder in folders:
                if os.path.isdir(data_dir + "/" + folder):
                    queue.append(data_dir + "/" + folder)

        return None



