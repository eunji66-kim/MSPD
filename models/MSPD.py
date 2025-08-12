# reference from https://github.com/wooseoklee4/AP-BSN

import torch

import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class MSPD(BaseModel):
    def __init__(self, in_channels=3, out_channels=3, base_channels=128, num_module1=9, num_module2=9, num_module3=9):
        super(MSPD, self).__init__()

        self.branch1 = DC_branchl(2, in_ch=in_channels, base_ch=base_channels, num_module=num_module1)
        self.branch2 = DC_branchl(2, in_ch=in_channels, base_ch=base_channels, num_module=num_module2)
        self.branch4 = DC_branchl(2, in_ch=in_channels, base_ch=base_channels, num_module=num_module3)

        ly = []
        ly += [ nn.Conv2d(in_channels*4, base_channels, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_channels, base_channels//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_channels//2, base_channels//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_channels//2, out_channels, kernel_size=1) ]
        self.tail = nn.Sequential(*ly)
    
    def forward(self, x):

        pd2 = pixel_shuffle_down_sampling(x, 2)
        pd4 = pixel_shuffle_down_sampling(x, 4)

        branch1 = self.branch1(x)
        branch2 = self.branch2(pd2)
        branch4 = self.branch4(pd4)

        ipd2 = pixel_shuffle_up_sampling(branch2, 2)
        ipd4 = pixel_shuffle_up_sampling(branch4, 4)

        concat2 = torch.cat([branch1, ipd2, ipd4, x], dim=1)
        output = self.tail(concat2)

        return output


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, base_ch, num_module):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        ly = []
        ly += [ CentralMaskedConv2d(base_ch, base_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, base_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(base_ch, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)

        return body


class DCl(nn.Module):
    def __init__(self, stride, base_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)
    

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x) 
        attention = self.sigmoid(attention)
        return x * attention
