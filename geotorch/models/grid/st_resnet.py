import torch
import torch.nn as nn
from collections import OrderedDict


## This implementation follows the implementation available here: https://github.com/BruceBinBoxing/ST-ResNet-Pytorch
class STResNet(nn.Module):
    '''
    Implementation of the model ST-ResNet. Paper link: https://dl.acm.org/doi/10.5555/3298239.3298479

    Parameters
    ..........
    c_conf (Tuple of 4 integers, Optional) - Default: (3, 2, 32, 32). Four integers denote closeness temporal length, no of channels/features, grid height, grid width.
    p_conf (Tuple of 4 integers, Optional) - Default: (3, 2, 32, 32). Four integers denote period temporal length, no of channels/features, grid height, grid width.
    t_conf (Tuple of 4 integers, Optional) - Default: (3, 2, 32, 32). Four integers denote trend temporal length, no of channels/features, grid height, grid width.
    external_dim (Int, Optional) - No of external dimensions. Default: 8
    nb_residual_unit (Int, Optional) - No of residual; units. Default: 3
    CF (Int, Optional) - No of Convolutional filter. Default: 64
    '''

    def __init__(self, c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32),
        t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3, CF=64):

        super(STResNet, self).__init__()
        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf
        self.CF = CF

        self.nb_flow, self.map_height, self.map_width = t_conf[1], t_conf[2], t_conf[3]

        self.relu = torch.relu
        self.tanh = torch.tanh
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.qr_nums = len(self.quantiles)

        if self.c_conf is not None:
            self.c_way = self._make_one_way(in_channels = self.c_conf[0] * self.c_conf[1], flow_num=self.c_conf[1])

        if self.p_conf is not None:
            self.p_way = self._make_one_way(in_channels = self.p_conf[0] * self.p_conf[1], flow_num=self.p_conf[1])

        if self.t_conf is not None:
            self.t_way = self._make_one_way(in_channels = self.t_conf[0] * self.t_conf[1], flow_num=self.t_conf[1])

        if self.external_dim != None and self.external_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.external_dim, 10, bias = True)),
                ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, self.nb_flow * self.map_height * self.map_width, bias = True)),
                ('relu2', nn.ReLU()),
            ]))

    def _make_one_way(self, in_channels, flow_num):

        return nn.Sequential(OrderedDict([
            ('conv1', _conv3x3(in_channels = in_channels, out_channels = self.CF)),
            ('ResUnits', _ResUnits(_Residual_unit, nb_filter = self.CF, repetations = self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('conv2', _conv3x3(in_channels = self.CF, out_channels = flow_num)),
            ('FusionLayer', _TrainableEltwiseLayer(n = flow_num, h = self.map_height, w = self.map_width))
        ]))

    def forward(self, input_c, input_p, input_t, input_ext = None):
        '''
        Parameters
        ..........
        input_c (Tensor) - Closeness sequence part of the input sample
        input_p (Tensor) - Period sequence part of the input sample
        input_t (Tensor) - Trend sequence part of the input sample
        input_ext (Tensor, Optional) - Extra input part of input sample when external_dim > 0. Default: None
        '''

        main_output = 0
        if self.c_conf is not None:
            input_c = input_c.view(-1, self.c_conf[0]*self.c_conf[1], self.c_conf[2], self.c_conf[3])
            out_c = self.c_way(input_c)
            main_output += out_c
        if self.p_conf is not None:
            input_p = input_p.view(-1, self.p_conf[0]*self.p_conf[1], self.p_conf[2], self.p_conf[3])
            out_p = self.p_way(input_p)
            main_output += out_p
        if self.t_conf is not None:
            input_t = input_t.view(-1, self.t_conf[0]*self.t_conf[1], self.t_conf[2], self.t_conf[3])
            out_t = self.t_way(input_t)
            main_output += out_t

        if self.external_dim != None and self.external_dim > 0 and input_ext != None:
            external_output = self.external_ops(input_ext)
            external_output = self.relu(external_output)
            external_output = external_output.view(-1, self.nb_flow, self.map_height, self.map_width)
            main_output += external_output


        main_output = self.tanh(main_output)

        return main_output



def _conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = "same")

class _Bn_relu_conv(nn.Module):
    def __init__(self, nb_filter):
        super(_Bn_relu_conv, self).__init__()
        self.batchNorm2d = nn.BatchNorm2d(num_features=nb_filter, eps=0.001, momentum=0.99, affine=False)
        self.relu = torch.relu
        self.conv1 = _conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        x = self.batchNorm2d(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x

class _Residual_unit(nn.Module):
    def __init__(self, nb_filter):
        super(_Residual_unit, self).__init__()
        self.bn_relu_conv1 = _Bn_relu_conv(nb_filter)
        self.bn_relu_conv2 = _Bn_relu_conv(nb_filter)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual

        return out

class _ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(_ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x

class _TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h, w):
        super(_TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w), requires_grad = True)

    def forward(self, x):
        x = x * self.weights

        return x



