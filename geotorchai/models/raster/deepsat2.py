import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import numpy as np


class DeepSatV2(nn.Module):
	'''
	Implementation of the classification model DeepSatV2. Paper link: https://arxiv.org/abs/1911.07747

	Parameters
	..........
	in_channels (Int) - Number of channels in the input images
	in_height (Int) - Height of the input images
	in_width (Int) - Width of the input images
	num_classes (Int) - Total number of classes/labels in the dataset
	num_filtered_features (Int) - Number of filtered features. Default: 0
	'''

	def __init__(self, in_channels, in_height, in_width, num_classes, num_filtered_features = 0):
		super(DeepSatV2, self).__init__()

		self.sequences_part1 = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3, padding="same"),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, padding="same"),
			nn.ReLU(),
			nn.ZeroPad2d((in_width//2, in_width//2, in_height//2, in_height//2)),
			nn.MaxPool2d(2),
			nn.Dropout(0.25))

		self.sequences_part2 = nn.Sequential(
			nn.Linear(64*in_height*in_width + num_filtered_features, 32),
			nn.BatchNorm1d(num_features=32, eps=0.001, momentum=0.99, affine=False),
			nn.ReLU(),
			nn.Linear(32, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, num_classes))


	def forward(self, images, filtered_features):
		'''
		Parameters
		..........
		images (Tensor) - Tensor containing the sample images
		filtered_features (Tensor) - Tensor containing the additional features
		'''

		x = self.sequences_part1(images)
		x = x.view(x.size(0), -1)

		if filtered_features != None:
			x = torch.cat((x, filtered_features), axis=1)

		x = self.sequences_part2(x)
		x = torch_f.softmax(x, dim=1)

		return x

