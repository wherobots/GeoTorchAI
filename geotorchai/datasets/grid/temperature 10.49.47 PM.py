
import os
from torch.utils.data import Dataset
from geotorchai.utility.exceptions import InvalidParametersException
from geotorchai.utility._download_utils import _download_cdsapi_files
import xarray as xr
import numpy as np


class Temperature(Dataset):
    '''
    This dataset is based on https://github.com/pangeo-data/WeatherBench

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


    def __init__(self, root, download=False, years=['2018'], pressure_level = '850', grid = [5.625,2.8125], lead_time = 2*24, normalize=True):
        super().__init__()

        self.all_months = ['01','02','03','04','05','06','07','08','09','10','11','12']
        self.all_days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
        '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
        self.all_times = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00',
        '11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']

        self.variable = 'temperature'
        self.level_type = 'pressure'
        self.pressure_level = pressure_level
        self.grid = grid
        self.product_type = 'reanalysis'
        self.format_name = 'netcdf'
        self.normalize = normalize

        if download:
            _download_cdsapi_files(root, self.variable, years, self.all_months, self.all_days, self.all_times, self.level_type, pressure_level = self.pressure_level, grid = self.grid, product_type = self.product_type, format_name = self.format_name)

        data_dir = self._get_path(root)
        arr = xr.open_mfdataset(f'{data_dir}/*.nc', combine='by_coords')
        self.full_data = arr['t'].values

        self.lead_time = lead_time
        self.use_lead_time = True
        self.sequential = False
        self.periodical = False

        self.timesteps = self.full_data.shape[0]
        self.grid_height = self.full_data.shape[1]
        self.grid_width = self.full_data.shape[2]

        self.full_data = self.full_data.reshape((self.timesteps, 1, self.grid_height, self.grid_width))

        max_data = np.max(self.full_data)
        min_data = np.min(self.full_data)
        self.min_max_diff = max_data - min_data
        if self.normalize:
            self.full_data = (2.0 * self.full_data - (max_data + min_data)) / (max_data - min_data)



    ## This method returns the total number of timesteps in the generated dataset
    def get_timesteps(self):
        return self.timesteps



    ## This method returns the height of the grid in the generated dataset
    def get_grid_height(self):
        return self.grid_height



    ## This method returns the width of the grid in the generated dataset
    def get_grid_width(self):
        return self.grid_width


    def get_min_max_difference(self):
        return self.min_max_diff



    def set_sequential_representation(self, history_length, prediction_length):
        self._generate_sequence_data(history_length, prediction_length)
        self.use_lead_time = False
        self.sequential = True
        self.periodical = False


    def set_periodical_representation(self, len_closeness = 3, len_period = 4, len_trend = 4, T_closeness=1, T_period=24, T_trend=24*7):
        self._generate_periodical_data(self.full_data, len_closeness, len_period, len_trend, T_closeness, T_period, T_trend)
        self.use_lead_time = False
        self.sequential = False
        self.periodical = True



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
        if self.use_lead_time:
            return len(self.full_data) - self.lead_time
        else:
            return len(self.Y_data)


    def __getitem__(self, index: int):
        if self.periodical:
            sample = {"x_closeness": self.X_closeness[index], \
                      "x_period": self.X_period[index], \
                      "x_trend": self.X_trend[index], \
                      "t_data": self.T_data[index], \
                      "y_data": self.Y_data[index]}
        else:
            if self.use_lead_time:
                x_data = self.full_data[index]
                y_data = self.full_data[index + self.lead_time]
            else:
                x_data = self.X_data[index]
                y_data = self.Y_data[index]
            sample = {"x_data": x_data, "y_data": y_data}

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


    def _generate_periodical_data(self, all_data, len_closeness, len_period, len_trend, T_closeness, T_period, T_trend):
        len_total,feature,map_height,map_width = all_data.shape

        time=np.arange(len_total,dtype=int)
        time_hour=time%T_period
        matrix_hour=np.zeros([len_total,24,map_height,map_width])
        for i in range(len_total):
            matrix_hour[i,time_hour[i],:,:]=1

        time_day=(time//T_period)%7
        matrix_day=np.zeros([len_total,7,map_height,map_width])
        for i in range(len_total):
            matrix_day[i,time_day[i],:,:]=1

        matrix_T=np.concatenate((matrix_hour,matrix_day),axis=1)

        if len_trend>0:
            number_of_skip_hours=T_trend*len_trend
        elif len_period>0:
            number_of_skip_hours=T_period*len_period
        elif len_closeness>0:
            number_of_skip_hours=T_closeness*len_closeness
        else:
            raise InvalidParametersException("Wrong parameters")

        Y=all_data[number_of_skip_hours:len_total]

        if len_closeness>0:
            self.X_closeness=all_data[number_of_skip_hours-T_closeness:len_total-T_closeness]
            for i in range(len_closeness-1):
                self.X_closeness=np.concatenate((self.X_closeness,all_data[number_of_skip_hours-T_closeness*(2+i):len_total-T_closeness*(2+i)]),axis=1)
        if len_period>0:
            self.X_period=all_data[number_of_skip_hours-T_period:len_total-T_period]
            for i in range(len_period-1):
                self.X_period=np.concatenate((self.X_period,all_data[number_of_skip_hours-T_period*(2+i):len_total-T_period*(2+i)]),axis=1)
        if len_trend>0:
            self.X_trend=all_data[number_of_skip_hours-T_trend:len_total-T_trend]
            for i in range(len_trend-1):
                self.X_trend=np.concatenate((self.X_trend,all_data[number_of_skip_hours-T_trend*(2+i):len_total-T_trend*(2+i)]),axis=1)

        matrix_T=matrix_T[number_of_skip_hours:]

        self.T_data = matrix_T
        self.Y_data = Y




