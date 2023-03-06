
import torch
import torch.nn as nn

## This implementation follows the implementation available here: https://github.com/pangeo-data/WeatherBench


class PeriodicalCNN(nn.Module):
	'''
    Implementation of the segmentation model Fully Convolutional Network (FCN). Paper link: https://arxiv.org/abs/1411.4038

    Parameters
    ..........
    num_features (Int) - Number of features or variables
    filters (List, Optional) - Each element represents the number of filters in a periodical convolution layer. Default: [32]
    kernels (List, Optional) - Each element represents the number of kernels in a periodical convolution layer. Default: [5]
    drop_val (Float, Optional) - Droput after a periodical convolution layer. Default: 0
    '''

	def __init__(self, num_features, filters=[32], kernels=[5], drop_val=0):
		super(PeriodicalCNN, self).__init__()

		if len(filters) != len(kernels):
			raise ValueError('Lengths of parameters filters and kernels should be same')
		if len(filters) == 0:
			raise ValueError('Parameter filters cannot be empty')

		moduleList = []

		input_channels = num_features
		for i in range(len(filters)):
			if i > 0:
				input_channels = filters[i-1]

			moduleList.append(_PeriodicConv2D(input_channels, filters[i], kernels[i]))
			moduleList.append(nn.ReLU())
			moduleList.append(nn.Dropout(drop_val))

		moduleList.append(_PeriodicConv2D(filters[-1], num_features, kernels[-1]))
		self.modelSequences = nn.Sequential(*moduleList)


	def forward(self, inputs):
		'''
        Parameters
        ..........
        inputs (Tensor) - Tensor containing the features
        '''

		return self.modelSequences(inputs)



class _PeriodicPadding2D(nn.Module):
	def __init__(self, pad_width):
		super(_PeriodicPadding2D, self).__init__()
		self.pad_width = pad_width

	def forward(self, x):
		if self.pad_width == 0:
			return x
		x = torch.cat((x[:, :, :, -self.pad_width:], x, x[:, :, :, :self.pad_width]), dim=3)
		x = nn.functional.pad(x, (0, 0, self.pad_width, self.pad_width, 0, 0, 0, 0))

		return x



class _PeriodicConv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size):
		super(_PeriodicConv2D, self).__init__()
		pad_width = (kernel_size - 1) // 2
		self.padding = _PeriodicPadding2D(pad_width)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding = "valid")

	def forward(self, x):
		return self.conv(self.padding(x))



