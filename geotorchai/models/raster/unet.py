import torch
import torch.nn as nn
import torch.nn.functional as torch_f

# This implementation is based on https://github.com/milesial/Pytorch-UNet
class UNet(nn.Module):
    '''
    Implementation of the segmentation model UNet. Paper link: https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

    Parameters
    ..........
    in_channels (Int) - Number of channels in the input images
    num_classes (Int) - Total number of output classes/channels
    '''

    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        self.input_conv = _ConvolutionNetwork(in_channels, 64)

        self.down_sample1 = _DownSampling(64, 128)
        self.down_sample2 = _DownSampling(128, 256)
        self.down_sample3 = _DownSampling(256, 512)
        self.down_sample4 = _DownSampling(512, 1024)

        self.up_sample1 = _UpSampling(1024, 512)
        self.up_sample2 = _UpSampling(512, 256)
        self.up_sample3 = _UpSampling(256, 128)
        self.up_sample4 = _UpSampling(128, 64)

        self.outpu_conv = _FinalConvolution(64, num_classes)

        
    def forward(self, images):
        '''
        Parameters
        ..........
        images (Tensor) - Tensor containing the sample images
        '''

        x1 = self.input_conv(images)

        x2 = self.down_sample1(x1)
        x3 = self.down_sample2(x2)
        x4 = self.down_sample3(x3)
        x5 = self.down_sample4(x4)

        x = self.up_sample1(x5, x4)
        x = self.up_sample2(x, x3)
        x = self.up_sample3(x, x2)
        x = self.up_sample4(x, x1)

        output = self.outpu_conv(x)
        return output



class _FinalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FinalConvolution, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv2d(x)



class _ConvolutionNetwork(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_ConvolutionNetwork, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_net(x)


class _DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_DownSampling, self).__init__()
        self.down_sampled_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            _ConvolutionNetwork(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_sampled_conv(x)


class _UpSampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_UpSampling, self).__init__()

        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.up_sample_cov = _ConvolutionNetwork(in_channels, out_channels)
            

    def forward(self, x1, x2):
        x1 = self.up_sample(x1)
        diff2 = x2.size()[2] - x1.size()[2]
        diff3 = x2.size()[3] - x1.size()[3]

        x1 = torch_f.pad(x1, [diff3//2, diff3 - diff3//2, diff2//2, diff2 - diff2//2])
        x = torch.cat([x2, x1], dim=1)
        return self.up_sample_cov(x)

