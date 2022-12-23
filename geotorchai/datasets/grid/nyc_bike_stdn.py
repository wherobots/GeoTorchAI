
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from geotorchai.utility._download_utils import _download_remote_file, _extract_archive


class BikeNYCSTDN(Dataset):
    '''
    This dataset is based on https://github.com/tangxianfeng/STDN/blob/master/file_loader.py
    Grid map_height and map_width = 10 and 20

    Parameters
    ..........
    root (String) - Path to the dataset if it is already downloaded. If not downloaded, it will be downloaded in the given path.
    download (Boolean, Optional) - Set to True if dataset is not available in the given directory. Default: False
    is_training_data (Boolean, Optional) - Set to True if you want to create the training dataset, False for testing dataset. Default: True
    att_lstm_num (Int, Optional) - Number of LSTM attributes. Default: 3
    long_term_lstm_seq_len (Int, Optional) - Length of long term LSTM sequence. Default: 3
    short_term_lstm_seq_len (Int, Optional) - Length of short term LSTM sequence. Default: 7
    hist_feature_daynum (Int, Optional) - Number of days in histogram feature. Default: 7
    last_feature_num (Int, Optional) - Default: 48
    nbhd_size (Int, Optional) - Default: 1
    cnn_nbhd_size (Int, Optional) - Default: 3
    '''

    DATA_URL = "https://raw.githubusercontent.com/tangxianfeng/STDN/master/data.zip"

    def __init__(self, root, download=False, is_training_data=True, att_lstm_num=3, long_term_lstm_seq_len=3, short_term_lstm_seq_len=7,
                 hist_feature_daynum=7, last_feature_num=48, nbhd_size=1, cnn_nbhd_size=3):
        super().__init__()

        if download:
            _download_remote_file(self.DATA_URL, root)
            _extract_archive(root + "/data.zip", root + "/data")

        data_dir = self._get_path(root)

        if is_training_data:
            flow_data = np.load(
                open(data_dir + "/bike_flow_train.npz", "rb"))["flow"] / 35.0
            volume_data = np.load(
                open(data_dir + "/bike_volume_train.npz", "rb"))["volume"] / 299.0
        else:
            flow_data = np.load(
                open(data_dir + "/bike_flow_test.npz", "rb"))["flow"] / 35.0
            volume_data = np.load(
                open(data_dir + "/bike_volume_test.npz", "rb"))["volume"] / 299.0

        self.timeslot_daynum = int(86400/1800)

        self._create_feature_vector(volume_data, flow_data, att_lstm_num, long_term_lstm_seq_len,
                                    short_term_lstm_seq_len, hist_feature_daynum, last_feature_num, nbhd_size, cnn_nbhd_size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):

        sample = {"att_cnnx": self.output_cnn_att_features[:, index, :, :, :], \
        "att_flow": self.output_flow_att_features[:, index, :, :, :], \
        "att_x": self.lstm_att_features[:, index, :, :], \
        "cnnx": self.cnn_features[:, index, :, :, :], \
        "flow": self.flow_features[:, index, :, :, :], \
        "x": self.short_term_lstm_features[:, index, :, :], \
        "label": self.labels[index]}

        return sample

    def _get_path(self, root_dir):
        queue = [root_dir]
        while queue:
            data_dir = queue.pop(0)
            folders = os.listdir(data_dir)
            if "bike_flow_train.npz" in folders and "bike_flow_test.npz" in folders and "bike_volume_train.npz" in folders and "bike_volume_test.npz" in folders:
                return data_dir

            for folder in folders:
                if os.path.isdir(data_dir + "/" + folder):
                    queue.append(data_dir + "/" + folder)

        return None

    # This is replication of sample_stdn method proposed by authors here: https://github.com/tangxianfeng/STDN/blob/master/file_loader.py
    def _create_feature_vector(self, data, flow_data, att_lstm_num, long_term_lstm_seq_len, short_term_lstm_seq_len, hist_feature_daynum, last_feature_num, nbhd_size, cnn_nbhd_size):
        cnn_att_features = []
        self.lstm_att_features = []
        flow_att_features = []
        for i in range(att_lstm_num):
            self.lstm_att_features.append([])
            cnn_att_features.append([])
            flow_att_features.append([])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i].append([])
                flow_att_features[i].append([])

        self.cnn_features = []
        self.flow_features = []
        for i in range(short_term_lstm_seq_len):
            self.cnn_features.append([])
            self.flow_features.append([])
        self.short_term_lstm_features = []
        self.labels = []

        time_start = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        time_end = data.shape[0]
        volume_type = data.shape[-1]

        for t in range(time_start, time_end):
            if t%100 == 0:
                print("Now sampling at {0} timeslots.".format(t))
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    
                    short_term_lstm_samples = []
                    for seqn in range(short_term_lstm_seq_len):
                        real_t = t - (short_term_lstm_seq_len - seqn)

                        cnn_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, volume_type))
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                        self.cnn_features[seqn].append(cnn_feature)

                        flow_feature_curr_out = flow_data[0, real_t, x, y, :, :]
                        flow_feature_curr_in = flow_data[0, real_t, :, :, x, y]
                        flow_feature_last_out_to_curr = flow_data[1, real_t - 1, x, y, :, :]
                        flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]

                        flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))
                        
                        flow_feature[:, :, 0] = flow_feature_curr_out
                        flow_feature[:, :, 1] = flow_feature_curr_in
                        flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                        local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4))
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                local_flow_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                        self.flow_features[seqn].append(local_flow_feature)

                        nbhd_feature = np.zeros((2*nbhd_size+1, 2*nbhd_size+1, volume_type))
                        for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                            for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                    continue
                                nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = data[real_t, nbhd_x, nbhd_y, :]
                        nbhd_feature = nbhd_feature.flatten()

                        last_feature = data[real_t - last_feature_num: real_t, x, y, :].flatten()

                        hist_feature = data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, x, y, :].flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature))
                        feature_vec = np.concatenate((feature_vec, nbhd_feature))

                        short_term_lstm_samples.append(feature_vec)
                    self.short_term_lstm_features.append(np.array(short_term_lstm_samples))

                    for att_lstm_cnt in range(att_lstm_num):
                        
                        long_term_lstm_samples = []
                        att_t = t - (att_lstm_num - att_lstm_cnt) * self.timeslot_daynum + (long_term_lstm_seq_len - 1) / 2 + 1
                        att_t = int(att_t)
                        for seqn in range(long_term_lstm_seq_len):
                            real_t = att_t - (long_term_lstm_seq_len - seqn)

                            cnn_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, volume_type))
                            for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            flow_feature_curr_out = flow_data[0, real_t, x, y, :, :]
                            flow_feature_curr_in = flow_data[0, real_t, :, :, x, y]
                            flow_feature_last_out_to_curr = flow_data[1, real_t - 1, x, y, :, :]
                            flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]

                            flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))
                            
                            flow_feature[:, :, 0] = flow_feature_curr_out
                            flow_feature[:, :, 1] = flow_feature_curr_in
                            flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                            flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                            local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4))
                            for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                                for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                    if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                        continue
                                    local_flow_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size), :] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, :]
                            flow_att_features[att_lstm_cnt][seqn].append(local_flow_feature)

                            nbhd_feature = np.zeros((2*nbhd_size+1, 2*nbhd_size+1, volume_type))
                            for nbhd_x in range(x - nbhd_size, x + nbhd_size + 1):
                                for nbhd_y in range(y - nbhd_size, y + nbhd_size + 1):
                                    if not (0 <= nbhd_x < data.shape[1] and 0 <= nbhd_y < data.shape[2]):
                                        continue
                                    nbhd_feature[nbhd_x - (x - nbhd_size), nbhd_y - (y - nbhd_size), :] = data[real_t, nbhd_x, nbhd_y, :]
                            nbhd_feature = nbhd_feature.flatten()

                            last_feature = data[real_t - last_feature_num: real_t, x, y, :].flatten()

                            hist_feature = data[real_t - hist_feature_daynum*self.timeslot_daynum: real_t: self.timeslot_daynum, x, y, :].flatten()

                            feature_vec = np.concatenate((hist_feature, last_feature))
                            feature_vec = np.concatenate((feature_vec, nbhd_feature))

                            long_term_lstm_samples.append(feature_vec)
                        self.lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))

                    self.labels.append(data[t, x , y, :].flatten())


        self.output_cnn_att_features = []
        self.output_flow_att_features = []
        for i in range(att_lstm_num):
            self.lstm_att_features[i] = np.array(self.lstm_att_features[i])
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])
                flow_att_features[i][j] = np.array(flow_att_features[i][j])
                self.output_cnn_att_features.append(cnn_att_features[i][j])
                self.output_flow_att_features.append(flow_att_features[i][j])
        
        for i in range(short_term_lstm_seq_len):
            self.cnn_features[i] = np.array(self.cnn_features[i])
            self.flow_features[i] = np.array(self.flow_features[i])

        self.output_cnn_att_features = torch.tensor(self.output_cnn_att_features)
        self.output_flow_att_features = torch.tensor(self.output_flow_att_features)
        self.lstm_att_features = torch.tensor(self.lstm_att_features)
        self.cnn_features = torch.tensor(self.cnn_features)
        self.flow_features = torch.tensor(self.flow_features)
        self.short_term_lstm_features = torch.tensor([self.short_term_lstm_features, ])
        self.labels = torch.tensor(self.labels)

