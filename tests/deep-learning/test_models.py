from geotorchai.datasets.raster import EuroSAT, Cloud38
from geotorchai.models.raster import DeepSatV2, SatCNN, FullyConvolutionalNetwork, UNet
from geotorchai.models.grid import STResNet, DeepSTN, ConvLSTM
from geotorchai.datasets.grid import BikeNYCDeepSTN
from torch.utils.data import DataLoader
import torch


class TestModels:


	def test_deepsatv2(self):
		model = DeepSatV2(3, 64, 64, 10, len(EuroSAT.ADDITIONAL_FEATURES))
		data = EuroSAT(root = "data/partial_datasets/raster1", bands=EuroSAT.RGB_BANDS, include_additional_features = True)
		loader = DataLoader(data, batch_size=2)
		inputs, labels, features = next(iter(loader)) 
		outputs = model(inputs, features)
		assert outputs.shape[0] == labels.shape[0]


	def test_satcnn(self):
		model = SatCNN(3, 64, 64, 10)
		data = EuroSAT(root = "data/partial_datasets/raster1", bands=EuroSAT.RGB_BANDS)
		loader = DataLoader(data, batch_size=2)
		inputs, labels = next(iter(loader)) 
		outputs = model(inputs)
		assert outputs.shape[0] == labels.shape[0]


	def test_fcn(self):
		model = FullyConvolutionalNetwork(4, 2)
		data = Cloud38(root = "data/partial_datasets/raster3")
		loader = DataLoader(data, batch_size=1)
		inputs, labels = next(iter(loader)) 
		outputs = model(inputs)
		predicted = outputs.argmax(dim=1)
		assert predicted.shape[0] == labels.shape[0]


	def test_unet(self):
		model = UNet(4, 2)
		data = Cloud38(root = "data/partial_datasets/raster3")
		loader = DataLoader(data, batch_size=1)
		inputs, labels = next(iter(loader)) 
		outputs = model(inputs)
		predicted = outputs.argmax(dim=1)
		assert predicted.shape[0] == labels.shape[0]


	def test_stresnet(self):
		len_closeness = 3
		len_period = 4
		len_trend = 4
		nb_residual_unit = 4
		map_height, map_width = 21, 12
		nb_flow = 2

		model = STResNet((len_closeness, nb_flow, map_height, map_width),
                     (len_period, nb_flow, map_height, map_width),
                     (len_trend, nb_flow , map_height, map_width),
                     external_dim = None, nb_residual_unit = nb_residual_unit)
		data = BikeNYCDeepSTN(root = "data/partial_datasets/grid1")
		loader = DataLoader(data, batch_size=4)

		sample = next(iter(loader))
		X_c = sample["x_closeness"].type(torch.FloatTensor)
		X_p = sample["x_period"].type(torch.FloatTensor)
		X_t = sample["x_trend"].type(torch.FloatTensor)
		Y_batch = sample["y_data"].type(torch.FloatTensor)

		outputs = model(X_c, X_p, X_t, None)
		assert outputs.shape[0] == Y_batch.shape[0]


	def test_deepstn(self):
		len_closeness = 3
		len_period = 4
		len_trend = 4
		nb_residual_unit = 4
		map_height, map_width = 21, 12
		nb_flow = 2

		model = DeepSTN(H=map_height, W=map_width,channel=2,
                          c=len_closeness,p=len_period, t = len_trend,
                          pre_F=64,conv_F=64,R_N=2,
                          is_plus=True,
                          plus=8,rate=1,
                          is_pt=True,P_N=9,T_F=56,PT_F=9,T=24,
                          dropVal=0.1)
		data = BikeNYCDeepSTN(root = "data/partial_datasets/grid1")
		loader = DataLoader(data, batch_size=2)

		sample = next(iter(loader))
		X_c = sample["x_closeness"].type(torch.FloatTensor)
		X_p = sample["x_period"].type(torch.FloatTensor)
		X_t = sample["x_trend"].type(torch.FloatTensor)
		t_data = sample["t_data"].type(torch.FloatTensor)
		p_data = sample["p_data"].type(torch.FloatTensor)
		Y_batch = sample["y_data"].type(torch.FloatTensor)

		outputs = model(X_c, X_p, X_t, t_data, p_data)
		assert outputs.shape[0] == Y_batch.shape[0]








