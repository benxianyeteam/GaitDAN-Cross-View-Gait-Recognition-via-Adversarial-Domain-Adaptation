import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import os
from torch.nn.parameter import Parameter


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

class LOCALBlock(nn.Module):
    def __init__(self, basicblock, p):
        super(LOCALBlock, self).__init__()
        self.block = basicblock
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

class TemporalBlock(nn.Module):
    def __init__(self, basicblock):
        super(TemporalBlock, self).__init__()
        self.block = basicblock
    def forward(self, x):
        x = self.block(x)
        return x

class MT3D3(nn.Module):
    '''
    上下两路：短时序+长时序
    '''
    def __init__(self, planes_list=[1, 32, 64, 128], bin_num=32, hidden_dim=256):
        super(MT3D3, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Conv3d(self.planes_list[0], self.planes_list[1], kernel_size=(3,3,3), stride=(1,1,1),padding=(1,1,1), bias=False)
        self.global1 = B3DBlock(self.planes_list[1], self.planes_list[2])
        self.global2 = B3DBlock(self.planes_list[2], self.planes_list[2])
        self.global3 = B3DBlock(self.planes_list[2], self.planes_list[3])

        self.temag1 = nn.Conv3d(self.planes_list[1], self.planes_list[1], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        self.temag2 = nn.Conv3d(self.planes_list[2], self.planes_list[2], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        self.temag3 = nn.Conv3d(self.planes_list[3], self.planes_list[3], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        '''下支'''
        self.local1 = TemporalBlock(B3DBlock(self.planes_list[1], self.planes_list[2]))
        self.local2 = TemporalBlock(B3DBlock(self.planes_list[2], self.planes_list[2]))
        self.local3 = TemporalBlock(B3DBlock(self.planes_list[2], self.planes_list[3]))

        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2*self.bin_num, 128, self.hidden_dim)))])
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
        # x [b ,c, t, h, w]
        x = x.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous()
        x = self.layer1(x)

        gl = F.max_pool3d(self.global1(x), kernel_size=(1, 2, 2), stride=(1, 2, 2))
        lo = F.max_pool3d(self.local1(self.temag1(x)), kernel_size=(1, 2, 2), stride=(1, 2, 2)) + self.temag2(gl)
        gl = self.global2(gl)
        gl = self.global3(gl)
        lo = self.local2(lo)
        lo = self.local3(lo) + self.temag3(gl)

        gl = torch.max(gl, 2)[0]  # frame_pooling
        lo = torch.max(lo, 2)[0]  # frame_pooling

        n, c, h, w = gl.size()
        gl = gl.view(n, c, self.bin_num, -1)
        gl = gl.mean(3) + gl.max(3)[0]
        lo = lo.view(n, c, self.bin_num, -1)
        lo = lo.mean(3) + lo.max(3)[0]
        feature = torch.cat((gl, lo), 2).permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin[0])  # [62,64,256]
        feature = feature.permute(1, 0, 2).contiguous()  # [64,62,256]
        del x, gl, lo
        return feature


class MT3D4(nn.Module):
    '''
    上下两路：短时序+长时序,训练大库
    '''
    def __init__(self, planes_list=[1, 32, 64, 128, 256], bin_num=32, hidden_dim=256):
        super(MT3D4, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Conv3d(self.planes_list[0], self.planes_list[1], kernel_size=(3,3,3), stride=(1,1,1),padding=(1,1,1), bias=False)
        self.global1 = B3DBlock(self.planes_list[1], self.planes_list[2])
        self.global2 = B3DBlock(self.planes_list[2], self.planes_list[2])
        self.global3 = B3DBlock(self.planes_list[2], self.planes_list[3])
        self.global4 = B3DBlock(self.planes_list[3], self.planes_list[3])
        self.global5 = B3DBlock(self.planes_list[3], self.planes_list[4])

        self.temag1 = nn.Conv3d(self.planes_list[1], self.planes_list[1], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        self.temag2 = nn.Conv3d(self.planes_list[2], self.planes_list[2], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        self.temag3 = nn.Conv3d(self.planes_list[3], self.planes_list[3], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        self.temag4 = nn.Conv3d(self.planes_list[4], self.planes_list[4], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        '''下支'''
        self.local1 = TemporalBlock(B3DBlock(self.planes_list[1], self.planes_list[2]))
        self.local2 = TemporalBlock(B3DBlock(self.planes_list[2], self.planes_list[2]))
        self.local3 = TemporalBlock(B3DBlock(self.planes_list[2], self.planes_list[3]))
        self.local4 = TemporalBlock(B3DBlock(self.planes_list[3], self.planes_list[3]))
        self.local5 = TemporalBlock(B3DBlock(self.planes_list[3], self.planes_list[4]))

        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2*self.bin_num, 256, self.hidden_dim)))])
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
        # x [b ,c, t, h, w]
        x = x.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous()
        x = self.layer1(x)

        gl = F.max_pool3d(self.global1(x), kernel_size=(1, 2, 2), stride=(1, 2, 2))
        lo = F.max_pool3d(self.local1(self.temag1(x)), kernel_size=(1, 2, 2), stride=(1, 2, 2)) + self.temag2(gl)
        gl = self.global2(gl)
        gl = self.global3(gl)
        lo = self.local2(lo)
        lo = self.local3(lo) + self.temag3(gl)
        gl = self.global4(gl)
        gl = self.global5(gl)
        lo = self.local4(lo)
        lo = self.local5(lo) + self.temag4(gl)


        gl = torch.max(gl, 2)[0]  # frame_pooling
        lo = torch.max(lo, 2)[0]  # frame_pooling

        n, c, h, w = gl.size()
        gl = gl.view(n, c, self.bin_num, -1)
        gl = gl.mean(3) + gl.max(3)[0]
        lo = lo.view(n, c, self.bin_num, -1)
        lo = lo.mean(3) + lo.max(3)[0]
        feature = torch.cat((gl, lo), 2).permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin[0])  # [62,64,256]
        feature = feature.permute(1, 0, 2).contiguous()  # [64,62,256]
        del x, gl, lo
        return feature

