import torch
import torch.nn as nn
import numpy as np


##This implementation follows the Keras implementation available here: https://github.com/FIBLAB/DeepSTN
class _Conv_unit0(nn.Module):
    def __init__(self, Fin, Fout, dropVal, H, W):
        super(_Conv_unit0, self).__init__()
        self.Fin = Fin
        self.H = H
        self.W = W
        self.relu = torch.relu
        self.batchNorm2d = nn.BatchNorm2d(num_features=Fin, eps=0.001, momentum=0.99, affine=False)
        self.dropout = nn.Dropout(dropVal)
        self.conv = nn.Conv2d(Fin, Fout, kernel_size=3, padding = "same")

    def forward(self, x):
        x = x.view(-1, self.Fin, self.H, self.W)
        x = self.relu(x)
        x = self.batchNorm2d(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

class _Conv_unit1(nn.Module):
    def __init__(self, Fin, Fout, dropVal, H, W):
        super(_Conv_unit1, self).__init__()
        self.Fin = Fin
        self.H = H
        self.W = W
        self.relu = torch.relu
        self.batchNorm2d = nn.BatchNorm2d(num_features=Fin, eps=0.001, momentum=0.99, affine=False)
        self.dropout = nn.Dropout(dropVal)
        self.conv = nn.Conv2d(Fin, Fout, kernel_size=1, padding = "same")

    def forward(self, x):
        x = x.view(-1, self.Fin, self.H, self.W)
        x = self.relu(x)
        x = self.batchNorm2d(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


## The following implementation of _Multiply class was proposed here: https://discuss.pytorch.org/t/how-to-create-a-_Multiply-layer-which-supports-backprop/56220
class _Multiply(nn.Module):
    def __init__(self):
        super(_Multiply, self).__init__()

    def forward(self, x):
        result = torch.ones(x[0].size()).to(self._device)

        for t in x:
            result *= t

        return t

    def _set_device(self, _device):
        self._device = _device



class _Res_plus(nn.Module):
    def __init__(self, F, Fplus, rate, dropVal, H, W):
        super(_Res_plus, self).__init__()
        self.F = F
        self.Fplus = Fplus
        self.rate = rate
        self.H = H
        self.W = W

        self.cl_conv1A = _Conv_unit0(F, F-Fplus, dropVal, H, W)
        self.cl_conv1B = nn.AvgPool2d(kernel_size=rate, stride=rate) ## there was padding = valid in keras version
        self.relu = torch.relu
        self.batchNorm2d = nn.BatchNorm2d(num_features=F, eps=0.001, momentum=0.99, affine=False)
        self.plus_conv = nn.Conv2d(F, Fplus*H*W, kernel_size=(int(np.floor(H/rate)),int(np.floor(W/rate))))
        self.cl_conv1 = _Conv_unit0(F, F, dropVal, H, W)

    def forward(self, x):
        x_org = x.view(-1, self.F, self.H, self.W)
        x2 = self.cl_conv1A(x_org)
        if self.rate !=1:
            x = self.cl_conv1B(x_org)
        else:
            x = x_org
        x = self.relu(x)
        x = self.batchNorm2d(x)
        x = self.plus_conv(x)
        x = x.view(-1, self.Fplus, self.H, self.W)
        x = torch.cat((x2, x), axis=1)
        x = self.cl_conv1(x)
        x = x_org + x

        return x


class _Res_normal(nn.Module):
    def __init__(self, F, dropVal, H, W):
        super(_Res_normal, self).__init__()
        self.F = F
        self.H = H
        self.W = W

        self.cl_conv1 = _Conv_unit0(F, F, dropVal, H, W)
        self.cl_conv2 = _Conv_unit0(F, F, dropVal, H, W)

    def forward(self, x):
        x_org = x.view(-1, self.F, self.H, self.W)
        x = self.cl_conv1 (x_org)
        x = self.cl_conv2 (x)
        x = x_org + x

        return x



class _T_trans(nn.Module):
    def __init__(self, T, T_F, H, W):
        super(_T_trans, self).__init__()
        self.T = T
        self.H = H
        self.W = W

        self.T_mid = nn.Conv2d(T+7, T_F, kernel_size=1, padding = "same")
        self.T_act = torch.relu
        self.T_fin = nn.Conv2d(T_F, 1, kernel_size=1, padding = "same")


    def forward(self, x):
        x = x.view(-1, self.T + 7, self.H, self.W)
        x = self.T_mid(x)
        x = self.T_act (x)
        x = self.T_fin (x)
        x = self.T_act (x)

        return x


class _PT_trans(nn.Module):
    def __init__(self, P_N, PT_F, T, T_F, H, W, isPT_F):
        super(_PT_trans, self).__init__()
        self.P_N = P_N
        self.T = T
        self.H = H
        self.W = W
        self.isPT_F = isPT_F

        self.t_trans = _T_trans(T, T_F, H, W)
        self.multiply = _Multiply()
        self.conv = nn.Conv2d(P_N, PT_F, kernel_size=1, padding = "same")

    def forward(self, x):
        poi_in = x[0].view(-1, self.P_N, self.H, self.W)
        time_in = x[1].view(-1, self.T+7, self.H, self.W)

        t_x = self.t_trans(time_in)
        if self.P_N >= 2:
            t_x = torch.cat(tuple([t_x]*self.P_N), axis=1)

        poi_time = self.multiply(torch.stack([poi_in, t_x]))
        if self.isPT_F:
            poi_time = self.conv(poi_time)

        return poi_time

    def _set_device(self, _device):
        self.multiply._set_device(_device)



#isPT_F decides whether PT_model uses one more Conv after _Multiplying PoI and Time, 1 recommended
class DeepSTN(nn.Module):
    def __init__(self, H=21, W=12, channel=2,\
        c=3, p=4, t=4,\
        pre_F=64, conv_F=64, R_N=2,\
        is_plus=True,\
        plus=8, rate=2,\
        is_pt=True,\
        P_N=6, T_F=28, PT_F=6, T=24,\
        dropVal=0,\
        kernel1=1,\
        isPT_F=1):
        super(DeepSTN, self).__init__()

        self._device = None

        self.H = H
        self.W = W
        self.T = T
        self.P_N = P_N
        self.is_pt = is_pt
        self.kernel1 = kernel1
        self.isPT_F = isPT_F
        self.is_plus = is_plus
        self.R_N = R_N

        self.channel_c = channel*c
        self.channel_p = channel*p
        self.channel_t = channel*t

        self.conv1 = nn.Conv2d(self.channel_c, pre_F, kernel_size=1, padding = "same")
        self.conv2 = nn.Conv2d(self.channel_p, pre_F, kernel_size=1, padding = "same")
        self.conv3 = nn.Conv2d(self.channel_t, pre_F, kernel_size=1, padding = "same")

        self.ptTrans = _PT_trans(P_N, PT_F, T, T_F, H, W, isPT_F)
        self.cpt1_0 = _Conv_unit1(pre_F*3+PT_F*isPT_F+P_N*(not isPT_F), conv_F, dropVal, H, W)
        self.cpt0_0 = _Conv_unit0(pre_F*3+PT_F*isPT_F+P_N*(not isPT_F), conv_F, dropVal, H, W)

        self.cpt1_1 = _Conv_unit1(pre_F*3, conv_F, dropVal, H, W)
        self.cpt0_1 = _Conv_unit0(pre_F*3, conv_F, dropVal, H, W)

        self.resPlus = _Res_plus(conv_F, plus, rate, dropVal, H, W)
        self.resNormal = _Res_normal(conv_F, dropVal, H, W)

        self.relu = torch.relu
        self.batchNorm2d = nn.BatchNorm2d(num_features=conv_F, eps=0.001, momentum=0.99, affine=False)
        self.dropout = nn.Dropout(dropVal)
        self.conv4 = nn.Conv2d(conv_F, channel, kernel_size=1, padding = "same")
        self.tanh = torch.tanh
        

    def forward(self, c_input, p_input, t_input, time_in = None, poi_in = None):
        if self._device == None:
            if c_input.get_device() == 0:
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
            self.ptTrans._set_device(self._device)

        c_input = c_input.view(-1, self.channel_c, self.H, self.W)
        p_input = p_input.view(-1, self.channel_p, self.H, self.W)
        t_input = t_input.view(-1, self.channel_t, self.H, self.W)

        c_out1 = self.conv1(c_input)
        p_out1 = self.conv2(p_input)
        t_out1 = self.conv3(t_input)

        if self.is_pt:
            time_in = time_in.view(-1, self.T+7, self.H, self.W)
            poi_in = poi_in.view(-1, self.P_N, self.H, self.W)
            poi_time = self.ptTrans([poi_in,time_in])
            cpt_con1 = torch.cat((c_out1, p_out1, t_out1, poi_time), axis=1)

            if self.kernel1:
                cpt = self.cpt1_0(cpt_con1)
            else:
                cpt = self.cpt0_0(cpt_con1)
        else:
            cpt_con1 = torch.cat((c_out1,p_out1,t_out1), axis=1)
            if self.kernel1:
                cpt = self.cpt1_1(cpt_con1)
            else:
                cpt = self.cpt0_1(cpt_con1)

        if self.is_plus:
            for i in range(self.R_N):
                cpt = self.resPlus(cpt)
        else:
            for i in range(self.R_N):
                cpt = self.resNormal(cpt)

        
        cpt_out = self.relu(cpt)
        cpt_out = self.batchNorm2d(cpt_out)
        cpt_out = self.dropout(cpt_out)
        cpt_out = self.conv4(cpt_out)
        cpt_out = self.tanh(cpt_out)

        return cpt_out

