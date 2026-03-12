import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.nn.parameter import Parameter
import cv2
import numpy as np
#from BERT import BERT5
#from .basic_blocks import SetBlock, BasicConv2d

import torch.nn as nn

def gem(x, p=6.5, eps=1e-6):
    # print('x-',x.shape)
    # print('xpow-',x.clamp(min=eps).pow(p).shape)
    # print(F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).shape)
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=6.5, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        # print('p-',self.p)
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem1(x, p=3.0, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM_1(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM_1,self).__init__()
        #self.p = Parameter(torch.ones(1)*p)
        self.p=1
        self.eps = eps
    def forward(self, x):
        return gem1(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p) + ', ' + 'eps=' + str(self.eps) + ')'

class Temporal(nn.Module):
    def __init__(self, inplanes, planes, bias=False, **kwargs):
        super(Temporal, self).__init__()

    def forward(self, x):
        
        out = torch.max(x, 2)[0]
        return out

class BasicConv3d_p(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, bias=False, p=8, FM=False, **kwargs):
        super(BasicConv3d_p, self).__init__()
        self.p = p
        self.fm = FM
        self.convdl = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
        self.convdg = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
    def forward(self, x):
        n, c, t, h, w = x.size()
        scale = h//self.p
        # print('p-',x.shape,n, c, t, h, w,'scale-',scale)
        feature = list()
        for i in range(self.p):
            temp = self.convdl(x[:,:,:,i*scale:(i+1)*scale,:])
            # print(temp.shape,i*scale,(i+1)*scale)
            feature.append(temp)

        outl = torch.cat(feature, 3)
        # print('outl-',outl.shape)
        outl = F.leaky_relu(outl, inplace=True)

        outg = self.convdg(x)
        outg = F.leaky_relu(outg, inplace=True)
        # print('outg-',outg.shape)
        if not self.fm:
            # print('1-1')
            out = outg + outl
        else:
            # print('1-2')
            out = torch.cat((outg, outl), dim=3)
        return out

class LocalConv3d_p(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, bias=False, p=2, **kwargs):
        super(LocalConv3d_p, self).__init__()
        self.block = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias,
                                padding=((kernel - 1) // 2, (kernel - 1) // 2, (kernel - 1) // 2))
        self.p = p
    def forward(self, x):
        b, c, t, h, w = x.size()
        scale = h//self.p
        feature = list()
        for i in range(self.p):
            temp = self.block(x[:,:,:,i*scale:(i+1)*scale,:])
            feature.append(temp)
        out = torch.cat(feature, 3)
        out = F.leaky_relu(out, inplace=True)
        del x, feature
        return out

class LocalSTAConv3d_p(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, bias=False, p=2, **kwargs):
        super(LocalSTAConv3d_p, self).__init__()
        self.block = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias,
                                padding=((kernel - 1) // 2, (kernel - 1) // 2, (kernel - 1) // 2))
        self.p = p
        self.STAlist = nn.ModuleList([STE2() for i in range(self.p)])
    def forward(self, x):
        b, c, t, h, w = x.size()
        scale = h//self.p
        feature = list()
        for i in range(self.p):
            x_input = x[:,:,:,i*scale:(i+1)*scale,:]
            x_input = x_input + x_input * self.STAlist[i](x_input)
            temp = self.block(x_input)
            feature.append(temp)
        out = torch.cat(feature, 3)
        out = F.leaky_relu(out, inplace=True)
        del x, feature, x_input
        return out

class BasicConv3d(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), bias=bias, dilation=(dilation, 1, 1), padding=(dilation, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, inplace=True)
        return out

class BasicConv3d1_1(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(BasicConv3d1_1, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), bias=bias, dilation=(dilation, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, inplace=True)
        return out

class LocaltemporalAG(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(LocaltemporalAG, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(3,1,1), bias=bias,padding=(0, 0, 0))

    def forward(self, x):
        out1 = self.conv1(x)
        out = F.leaky_relu(out1, inplace=True)
        return out

class B3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(B3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.conv3 = nn.Conv3d(in_planes, out_planes, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False)

    def forward(self, x):
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = F.leaky_relu(x,inplace=True)
        return x


class SeparateBNNecks(nn.Module):
    """
        GaitSet: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=False, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num                  #31
        self.class_num = class_num          #74
        self.norm = norm                    #True
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))   #[31, 256, 74]
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)    #BN [31*256]
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.parallel_BN1d:
            p, n, c = x.size()
            x = x.transpose(0, 1).contiguous().view(n, -1)  # [n, p*c]
            x = self.bn1d(x)
            x = x.view(n, p, c).permute(1, 0, 2).contiguous() #[p, n, c]
        else:
            x = torch.cat([bn(_.squeeze(0)).unsqueeze(0)
                           for _, bn in zip(x.split(1, 0), self.bn1d)], 0)  # [p, n, c]
        if self.norm:
            feature = F.normalize(x, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            feature = x
            logits = feature.matmul(self.fc_bin)
        return feature, logits


class C3DVGG(nn.Module):

    def __init__(self, num_classes=74):
        super(C3DVGG, self).__init__()
        _set_channels = [32, 64, 128]

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])

        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2], FM=True)
        self.fpb3d = Temporal(_set_channels[2], _set_channels[2])

        self.Gem = GeM()

        self.bin_numgl = [32 * 2]
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(sum(self.bin_numgl), _set_channels[2], _set_channels[2])))

        self.relu = nn.ReLU()
        for m in self.modules():
            # print('---')
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        n, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, 0:1, :, :]), dim=1)
        # print(x.shape)
        x = x.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous()
        '''
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, :, 0:1, :, :]), dim=2)
        # print(x.shape)
        '''
        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        # print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        # print('pool2d2-',x2d.shape)
        x2da3d = self.conv2dlayer3a_3d(x2d)
        # print('conv2dlayer3a_3d-',x2da3d.shape)
        x2db3d = self.conv2dlayer3b_3d(x2da3d)
        # print('conv2dlayer3b_3d-',x2db3d.shape)

        x2db3d = self.fpb3d(x2db3d)
        # print('x2db-',x2db3d.shape)

        # xgem = self.Gem(x2db3d)
        # print('xgem-',xgem.shape)

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z = z.mean(3) + z.max(3)[0]
            z = self.Gem(z).squeeze(-1)
            feature.append(z)
        feature = torch.cat(feature, 2)
        feature = feature.permute(2, 0, 1).contiguous()
        # print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin[0])  # [64,4,128]
        feature = feature.permute(1, 0, 2).contiguous()
        return feature, None

class C3DVGGcom(nn.Module):

    def __init__(self, num_classes=74):
        super(C3DVGGcom, self).__init__()
        _set_channels = [32, 64, 128]

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])

        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1], p=8)
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2], p=8)
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2], p=8, FM=True)
        self.fpb3d = Temporal(_set_channels[2], _set_channels[2])

        self.Gem = GeM()

        self.bin_numgl = [32*2]
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(sum(self.bin_numgl), _set_channels[2], _set_channels[2])))

        self.bn2 = nn.BatchNorm1d(_set_channels[2])
        self.fc2 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((sum(self.bin_numgl)), _set_channels[2], num_classes)))

        self.relu = nn.ReLU()
        for m in self.modules():
            # print('---')
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        n, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, 0:1, :, :]), dim=1)
        # print(x.shape)
        x = x.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous()
        '''
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x,x[:,:,0:1,:,:]),dim=2)
        # print(x.shape)
        '''
        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        # print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        # print('pool2d2-',x2d.shape)
        x2da3d = self.conv2dlayer3a_3d(x2d)
        # print('conv2dlayer3a_3d-',x2da3d.shape)
        x2db3d = self.conv2dlayer3b_3d(x2da3d)
        # print('conv2dlayer3b_3d-',x2db3d.shape)

        x2db3d = self.fpb3d(x2db3d)
        # print('x2db-',x2db3d.shape)

        # xgem = self.Gem(x2db3d)
        # print('xgem-',xgem.shape)

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z = z.mean(3) + z.max(3)[0]
            z = self.Gem(z).squeeze(-1)
            feature.append(z)
        feature = torch.cat(feature, 2)  #[n,c,h]
        feature = feature.permute(2, 0, 1).contiguous() #[h,n,c]
        # print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin[0])            #[64,4,128]
        feature = feature.permute(1, 2, 0).contiguous() #[n,c,h]
        # print('feature',feature.shape)

        featurebn = self.bn2(feature)                       #[4,128,64]
        # print('bn2-',featurebn2.shape)

        feature = featurebn.permute(2, 0, 1).contiguous()   #[64,4,128]
        # print('feature2-',feature.shape)
        feature = feature.matmul(self.fc2[0])               #[64,4,74]
        feature = feature.permute(1, 0, 2).contiguous()     #[4,64,74]
        
        # print('feature-',feature.shape)

        return featurebn,feature


class C3DVGGcomGRL(nn.Module):

    def __init__(self, num_classes=74):
        super(C3DVGGcomGRL, self).__init__()
        _set_channels = [32, 64, 128]
        self.ReverseLayer = GRL()
        self.Temperature = 1.0  ####

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])

        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2], FM=True)
        self.fpb3d = Temporal(_set_channels[2], _set_channels[2])

        self.Gem = GeM()

        self.bin_numgl = [32 * 2]
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(sum(self.bin_numgl), _set_channels[2], _set_channels[2])))

        self.bn2 = nn.BatchNorm1d(_set_channels[2])
        self.fc2 = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros((sum(self.bin_numgl)), _set_channels[2], num_classes)))

        self.relu = nn.ReLU()
        for m in self.modules():
            # print('---')
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        n, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, 0:1, :, :]), dim=1)
        # print(x.shape)
        x = x.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous()
        '''
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x,x[:,:,0:1,:,:]),dim=2)
        # print(x.shape)
        '''
        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        # print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        # print('pool2d2-',x2d.shape)
        x2da3d = self.conv2dlayer3a_3d(x2d)
        # print('conv2dlayer3a_3d-',x2da3d.shape)
        x2db3d = self.conv2dlayer3b_3d(x2da3d)
        # print('conv2dlayer3b_3d-',x2db3d.shape)

        x2db3d = self.fpb3d(x2db3d)
        # print('x2db-',x2db3d.shape)

        # xgem = self.Gem(x2db3d)
        # print('xgem-',xgem.shape)

        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z = z.mean(3) + z.max(3)[0]
            z = self.Gem(z).squeeze(-1)
            feature.append(z)
        feature = torch.cat(feature, 2)  #[n,c,h]
        feature = feature.permute(2, 0, 1).contiguous() #[h,n,c]
        # print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin[0])  # [64,4,128]
        feature = feature.permute(1, 2, 0).contiguous() #[n,c,h]
        # print('feature',feature.shape)

        featurebn = self.bn2(feature)  # [4,128,64]

        # ---------cse_fea----------
        feature = featurebn.permute(2, 0, 1).contiguous()  # [64,N,128]
        feature = feature.matmul(self.fc2[0])  # [64,N,74]
        feature = feature.permute(1, 0, 2).contiguous()  # [N,64,74]
        # ---------grl_fea----------
        fea_grl = F.normalize(featurebn, p=2, dim=2)
        fea_grl = fea_grl.permute(2, 0, 1).contiguous()  # [64,N,128]
        fea_grl = self.ReverseLayer(fea_grl)
        fea_grl = fea_grl.matmul(self.fc_view[0])  # [64,N,11]
        fea_grl = fea_grl / self.Temperature
        fea_grl = fea_grl.permute(1, 0, 2).contiguous()  # [N,64,11]

        return featurebn, feature, fea_grl


class B3DSTA1Block(nn.Module):
    def __init__(self, in_planes, out_planes, STABlock):
        super(B3DSTA1Block, self).__init__()
        self.attn = STABlock
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.conv3 = nn.Conv3d(in_planes, out_planes, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False)
        self.adjust = nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1),stride=(1,1,1), padding=(0,0,0), bias=False)
    def forward(self, x_raw):
        #a = self.attn(x_raw)[1]
        #b = self.attn(x_raw)[2]
        #x = x_raw + x_raw*self.attn(x_raw)[0]
        x = x_raw + x_raw * self.attn(x_raw)
        x_raw = self.adjust(x_raw)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = x_raw + x  ###残差链接
        x = F.leaky_relu(x,inplace=True)
        return x #, a, b


class STE2(nn.Module):
    '''3d卷积的时空注意力,空间注意力使用mean，这个好'''
    def __init__(self):
        super(STE2, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,1), padding=1, bias=False)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False)
    def forward(self, x):
        #x [b,c,t,h,w]
        b,c,t,h,w = x.shape
        x_s = x.mean(1, keepdim=True)
        x_s = x_s.mean(2)
        x_score1 = self.conv1(x_s)
        x_score1 = x_score1.unsqueeze(2)

        x_t = x.mean(1)
        x_t = F.avg_pool2d(x_t, x_t.size()[2:])
        x_t = x_t.unsqueeze(1)
        x_score2 = self.conv2(x_t)
        x_score = x_score1*x_score2
        x_score = torch.sigmoid(x_score)

        del x_s,x_t,x_score2, x_score1
        return x_score

#------------------------------------------梯度逆转---------------------------------------------
#----------------------------------------------------------------------------------------------
class ReverseF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, iter_num):
        output = input * 1.0
        ctx.iter_num = iter_num
        #ctx.save_for_backward(iter_num)
        return output

    @staticmethod
    def backward(ctx, gradOutput, high=1.0, low=0.0, alpha=10, max_iter=2*10000000.0):
        #iter_num = ctx.saved_tensors
        iter_num = ctx.iter_num
        if iter_num % 1000 == 0:
            print('back:',iter_num)
        ctx.coeff = np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num/ max_iter)) - (high - low) + low)
        return -ctx.coeff * gradOutput, None

class ReverseF1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, iter_num):
        output = input * 1.0
        ctx.iter_num = iter_num
        #ctx.save_for_backward(iter_num)
        return output

    @staticmethod
    def backward(ctx, gradOutput, high=1.0, low=0.0, alpha=10, max_iter=2*10000000.0):
        #iter_num = ctx.saved_tensors
        iter_num = ctx.iter_num
        if iter_num % 1000 == 0:
            print('back:',iter_num)
        #ctx.coeff = np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num/ max_iter)) - (high - low) + low)
        ctx.coeff = 0.1  #0.01  0.005
        return -ctx.coeff * gradOutput, None

class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()
        self.iter_num = 0
        #print('GRL-1:', self.iter_num)
    def forward(self, x):
        self.iter_num += 1
        #print('GRL-2:', self.iter_num)
        return ReverseF1.apply(x, self.iter_num)


class RandomErasing(nn.Module):
    def __init__(self, p=0.5, h=4):
        super(RandomErasing, self).__init__()
        self.p = p
        self.h = h

    def forward(self, x):
        rp = random.random()
        if rp > self.p:
            return x
        else:
            while True:
                he = random.randint(0, x.size()[3])
                if he + self.h <= x.size()[3]:
                    x[:,:,:,he:he+self.h,:] = 0.0
                    return x


class GL2GRL(nn.Module):
    '''
    global+local(2 block),hidden_dim=128,mean+max()
    '''
    def __init__(self, planes_list=[1, 32, 64, 128], bin_num=32, hidden_dim=128,num_classes=11):
        super(GL2GRL, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num
        self.hidden_dim = hidden_dim
        self.layer1 = BasicConv3d(planes_list[0], planes_list[1], kernel=3)
        self.layer2 = LocaltemporalAG(planes_list[1], planes_list[1])
        self.glo1 = BasicConv3d(self.planes_list[1], self.planes_list[2], kernel=3)
        self.glo2 = BasicConv3d(self.planes_list[2], self.planes_list[3], kernel=3)
        self.glo3 = BasicConv3d(self.planes_list[3], self.planes_list[3], kernel=3)

        self.loc1 = LocalConv3d_p(self.planes_list[1], self.planes_list[2], p=2)
        self.loc2 = LocalConv3d_p(self.planes_list[2], self.planes_list[3], p=2)
        self.loc3 = LocalConv3d_p(self.planes_list[3], self.planes_list[3], p=2)

        #self.Gem = GeM()
        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2*self.bin_num, 128, self.hidden_dim)))])

        self.ReverseLayer = GRL()
        #self.avgpool = GeM_1()
        self.bn2 = nn.BatchNorm2d(planes_list[3])
        self.cls = nn.ModuleList([nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, num_classes)
        ) for i in range(2*self.bin_num)])


        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # x [b ,c, t, h, w]
        # print(x.shape)
        n, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, 0:1, :, :]), dim=1)
        # print(x.shape)
        x = x.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous()
        n, c, t, h, w = x.size()
        # -------------------------------------------------------------
        x = self.layer1(x)
        x = self.layer2(x)

        gl = F.max_pool3d(self.glo1(x), kernel_size=(1, 2, 2), stride=(1, 2, 2))
        lo = F.max_pool3d(self.loc1(x), kernel_size=(1, 2, 2), stride=(1, 2, 2)) + gl
        gl = self.glo2(gl)
        lo = self.loc2(lo) + gl
        gl = self.glo3(gl)
        lo = self.loc3(lo) + gl

        gl = torch.max(gl, 2)[0]  # frame_pooling
        lo = torch.max(lo, 2)[0]  # frame_pooling
        feature = torch.cat((gl, lo), 2)   #[n, c, 2*h, w]

        n, c, h, w = feature.size()
        fea = feature.view(n, c, 2*self.bin_num, -1)
        fea = fea.max(3)[0]+fea.mean(3)
        fea = fea.permute(2, 0, 1).contiguous()
        fea = fea.matmul(self.fc_bin[0])  # [64,N,128]
        fea = fea.permute(1, 2, 0).contiguous()  # [N,128,64]

        feares = self.ReverseLayer(feature)       #梯度反转
        feares = self.bn2(feares)
        #feares = self.avgpool(feares)
        feares = feares.max(3)[0] + feares.mean(3)
        feares_new = []
        for i in range(2*self.bin_num):
            f = self.cls[i](feares[:,:,i]).unsqueeze(1)
            feares_new.append(f)
        feares_new = torch.cat(feares_new,1)
        del x, gl, lo, feature, feares
        return fea,feares_new

class GMl4STA3glgemcomGRL(nn.Module):
    '''
    global,multi-level注意力用STE2()，hidden_dim=128,mean+max(),fc后再接fc-relu-fc学每个part的score，最开始的提出的模型
    '''
    def __init__(self, planes_list=[1, 32, 64, 128], bin_num=32, hidden_dim=128,num_classes=74):
        super(GMl4STA3glgemcomGRL, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num
        self.hidden_dim = hidden_dim
        self.Gem = GeM()
        self.ReverseLayer = GRL()
        self.Temperature = 1.0   ####
        self.bin_numgl = [1, 32]
        bin_num_all = 0
        for n in self.bin_numgl:
            bin_num_all = bin_num_all + n
        self.bin_num_all = bin_num_all
        self.layer1 = BasicConv3d(planes_list[0], planes_list[1], kernel=3)
        self.layer2 = LocaltemporalAG(planes_list[1], planes_list[1])                #32
        self.glo1 = BasicConv3d(self.planes_list[1], self.planes_list[2], kernel=3)  #64
        self.glo2 = BasicConv3d(self.planes_list[2], self.planes_list[3], kernel=3)  #128
        self.glo3 = BasicConv3d(self.planes_list[3], self.planes_list[3], kernel=3)  #128

        '''注意力'''
        self.attn1 = STE2()
        self.attn2 = STE2()
        self.attn3 = STE2()
        self.cross_param = nn.Parameter(torch.ones(1, 2, 2), requires_grad=True)
        self.cross_param[:, :, 1].data.fill_(0)

        #self.lateral0 = BasicConv3d(self.planes_list[1], self.planes_list[2], kernel=1) #32->64
        self.lateral1 = BasicConv3d(self.planes_list[2], self.planes_list[3], kernel=3) #64->128
        self.lateral2 = BasicConv3d(self.planes_list[3], self.planes_list[3], kernel=3) #128->128
        #self.attn6 = STE2()
        #self.Gem = GeM()
        self.channel_num = planes_list[3]+planes_list[3]+planes_list[3]
        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1*self.bin_num_all , self.channel_num, self.hidden_dim)))])
        self.bn2 = nn.BatchNorm1d(planes_list[3])
        self.fc2 = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1 * self.bin_num_all ), 128, num_classes)))])
        self.fc_view= nn.ParameterList(
            [nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1 * self.bin_num_all), 128, 11)))])   ###gai

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
        # x [b ,c, t, h, w]
        # print(x.shape)
        n, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, 0:1, :, :]), dim=1)
        # print(x.shape)
        x = x.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous()
        n, c, t, h, w = x.size()
        # -------------------------------------------------------------
        x0 = self.layer1(x)
        x0 = self.layer2(x0)

        x1 = F.max_pool3d(self.glo1(x0), kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x1att = x1
        x1att = x1att + x1att * self.attn1(x1att)     ##
        later1 = self.lateral1(x1att)

        x2 = self.glo2(x1)
        x2att = self.apply_cross_stitch(x2, later1, self.cross_param[0][0])
        #x2att = x2
        x2att = x2att + x2att*self.attn2(x2att)  ##
        later2 = self.lateral2(x2att)

        x3 = self.glo3(x2)
        x3att = self.apply_cross_stitch(x3, later2, self.cross_param[0][1])
        #x3att = x3
        x3att = x3att + x3att*self.attn3(x3att)  ##

        gl = torch.cat((later1, later2, x3att), 1)
        gl = torch.max(gl, 2)[0]  # frame_pooling
        n, c, h, w = gl.size()
        # ----------------new-----------------
        gl_all = []
        for num_bin in self.bin_numgl:
            z = gl.view(n, c, num_bin, -1).contiguous()
            # z = z.mean(3) + z.max(3)[0]
            # z = z.max(3)[0]
            z = self.Gem(z).squeeze(-1)
            gl_all.append(z)
        gl = torch.cat(gl_all, 2)

        # ---------tri_fea----------
        feature = gl.permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin[0])  # [62,64,256]
        feature = feature.permute(1, 2, 0).contiguous()  # [N,128,64]
        featurebn = self.bn2(feature)
        # ---------cse_fea----------
        feature = featurebn.permute(2, 0, 1).contiguous()  # [64,N,128]
        feature = feature.matmul(self.fc2[0])  # [64,N,74]
        feature = feature.permute(1, 0, 2).contiguous()  # [N,64,74]
        # ---------grl_fea----------
        fea_grl = F.normalize(featurebn, p=2, dim=2)
        fea_grl = fea_grl.permute(2, 0, 1).contiguous()  # [64,N,128]
        fea_grl = self.ReverseLayer(fea_grl)
        fea_grl = fea_grl.matmul(self.fc_view[0])   # [64,N,11]
        fea_grl = fea_grl / self.Temperature
        fea_grl = fea_grl.permute(1, 0, 2).contiguous()  #[N,64,11]

        del  x0, gl, x1, x2, x3, x1att, x2att, x3att, later1, later2
        x = x.squeeze(1)
        n,t,h,w = x.size()
        if t>30:
            x = x[:,0:30,:,:]
        else:
            x = x.repeat(1,30,1,1)
            x = x[:,0:30,:,:]
        x = x.view(n,-1)

        return featurebn, feature, fea_grl,x

    def apply_cross_stitch(self, input1, input2, cross_param):
        cross_param = F.softmax(cross_param, dim=0)
        # print(cross_param)
        # output1 = cross_param[0] * input1 + cross_param[1] * input2
        output2 = cross_param[0] * input2 + cross_param[1] * input1 #[1*(),0*()]
        # output2 = 0 * input2 +1 * input1
        return output2



def params_count(net):
    list1 = []
    for p in net.parameters():
        # print('p-',p.shape)
        list1.append(p)
    # print('len(net.parameters)',len(list1))
    n_parameters = sum(p.numel() for p in net.parameters())
    print('-----Model param: {:.5f}M'.format(n_parameters / 1e6))
    # print('-----Model memory: {:.5f}M'.format(n_parameters/1e6))
    return n_parameters


def c3d_vgg_Fusion(**kwargs):
    """Constructs a ResNet-101 model.
    """
    #model = GMl4STA3glgemcomGRLbigv7(**kwargs)
    #model = GMl4STA3com(**kwargs)
    #model = GMl4STA3FFTv1(**kwargs)
    #model = GlobalSTA31FFT(**kwargs)
    #model = SetNetraw(**kwargs)
    #model = GMl4STA3glgemcomGRLbigv7(**kwargs)
    model = GMl4STA3glgemcomGRL(**kwargs)
    #model = C3DVGGcom(**kwargs)
    return model


if __name__ == "__main__":
    net = c3d_vgg_Fusion(num_classes=74).cuda()
    #net = c3d_vgg_Fusion().cuda()
    print(params_count(net))
    with torch.no_grad():
        x = torch.ones(4*30*64*44).reshape(4,30,64,44).cuda()
        #view = torch.ones(4).long().cuda()
        #view[2] = 2
        #print(view.shape)
        #x = torch.ones(4 * 3 * 32 * 64 * 44).reshape(4, 3, 32, 64, 44).cuda()
        # a = Variable(a.cuda)
        print('x=', x.shape)
        #print('view=', view.shape)
        a,b = net(x)
        #print('a,b=',a.shape,b.shape)
        #a,b,c= net(x)
        print('a', a.shape)
        print('b', b.shape)
        #print('c', c.shape)
