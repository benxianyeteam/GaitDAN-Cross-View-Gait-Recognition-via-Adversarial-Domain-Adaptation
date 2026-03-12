import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.nn.parameter import Parameter

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

class Temporal(nn.Module):
    def __init__(self, inplanes, planes, bias=False, **kwargs):
        super(Temporal, self).__init__()

    def forward(self, x):
        
        out = torch.max(x, 2)[0]
        return out


class BasicConv3d_p(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, bias=False, p=2, FM=False, **kwargs):
        super(BasicConv3d_p, self).__init__()
        self.p = p
        self.fm = FM
        self.convdl = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
        self.convdg = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
    def forward(self, x):
        n, c, t, h, w = x.size()
        scale = h//self.p
        feature = list()
        for i in range(self.p):
            temp = self.convdl(x[:,:,:,i*scale:(i+1)*scale,:])
            feature.append(temp)

        outl = torch.cat(feature, 3)
        outl = F.leaky_relu(outl, inplace=True)

        outg = self.convdg(x)
        outg = F.leaky_relu(outg, inplace=True)
        if not self.fm:
            out = outg + outl
        else:
            out = torch.cat((outg, outl), dim=3)
        return out


class BasicConv3d(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), bias=bias, dilation=(dilation, 1, 1), padding=(dilation, 1, 1))

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

class C3D_VGG(nn.Module):

    def __init__(self, num_classes=74):
        super(C3D_VGG, self).__init__()
        _set_channels = [32, 64, 128, 256]

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])


        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer25a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer25b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2])

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[2], _set_channels[3])
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[3], _set_channels[3], FM=True)

        self.fpb3d = Temporal(_set_channels[3], _set_channels[3])

        self.Gem = GeM()


        self.bin_numgl = [32*2]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[3], _set_channels[3])))
                    ])
                



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
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x,x[:,:,0:1,:,:]),dim=2)
        # print(x.shape)

        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        # print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        # print('pool2d2-',x2d.shape)

        x2d = self.conv2dlayer25a_3d(x2d)
        x2d = self.conv2dlayer25b_3d(x2d)

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
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            z2 = self.Gem(z).squeeze(-1)
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        # print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 2, 0).contiguous()
        # print('feature',feature.shape)

        return feature,None


class tryNetgl(nn.Module):

    def __init__(self, num_classes=74):
        super(tryNetgl, self).__init__()
        _set_channels = [32, 64, 128, 256]

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])

        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer25a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[2], _set_channels[2], FM=True)

        self.fpb3d = Temporal(_set_channels[2], _set_channels[2])

        self.Gem = GeM()

        self.bin_numgl = [32 * 2]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[2], _set_channels[2])))
        ])

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
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, :, 0:1, :, :]), dim=2)
        # print(x.shape)

        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        x2d = self.pool2d1(x2d)
        x2d = self.conv2dlayer2a(x2d)
        x2d = self.pool2d2(x2d)

        x2d = self.conv2dlayer25a_3d(x2d)
        #x2d = self.conv2dlayer25b_3d(x2d)

        x2da3d = self.conv2dlayer3a_3d(x2d)
        #x2da3d = self.conv2dlayer3b_3d(x2da3d)

        x2db3d = self.fpb3d(x2da3d)
        _, c2d, _, _ = x2db3d.size()

        feature = list()
        for num_bin in self.bin_numgl:
            z = x2db3d.view(n, c2d, num_bin, -1).contiguous()
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            z2 = self.Gem(z).squeeze(-1)
            # print('z2-',z2.shape)
            feature.append(z2)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        # print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 2, 0).contiguous()
        # print('feature',feature.shape)

        return feature, None


###################################################basic module##############################################
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

class MultiSpan(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,  **kwargs):
        super(MultiSpan, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, bias=False, **kwargs)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=2, dilation=2, bias=False, **kwargs)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=4, dilation=4, bias=False, **kwargs)
        #self.bn1 = nn.BatchNorm1d(out_channels)
        #self.bn2 = nn.BatchNorm1d(out_channels)
        #self.bn3 = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, c, h, w)
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1,x2,x3), 1)
        x = x.view(n, 3*c, h, w)
        x = x.view(b, -1, 3*c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        del x1,x2
        return x



###################################################models####################################################
class GaitGL(nn.Module):

    def __init__(self, num_classes=74):
        super(GaitGL, self).__init__()
        _set_channels = [32, 64, 128, 256]

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d(1, _set_channels[0], kernel=3)
        # self.conv2dlayer1b = BasicConv3d(_set_channels[0], _set_channels[0])
        self.pool2d1 = LocaltemporalAG(_set_channels[0], _set_channels[0])

        self.conv2dlayer2a = BasicConv3d_p(_set_channels[0], _set_channels[1])
        # self.conv2dlayer2b = BasicConv3d(_set_channels[1], _set_channels[1])
        self.pool2d2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        #self.conv2dlayer25a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        #self.conv2dlayer25b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2],)

        self.conv2dlayer3a_3d = BasicConv3d_p(_set_channels[1], _set_channels[2])
        self.conv2dlayer3b_3d = BasicConv3d_p(_set_channels[2], _set_channels[2], FM=True)

        self.fpb3d = Temporal(_set_channels[2], _set_channels[2])

        self.Gem = GeM()

        self.bin_numgl = [32 * 2]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_numgl), _set_channels[2], _set_channels[2])))
        ])

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
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, :, 0:1, :, :]), dim=2)
        # print(x.shape)

        # ----------------2d--------------------
        x2d = self.conv2dlayer1a(x)
        # x2d = self.conv2dlayer1b(x2d)
        x2d = self.pool2d1(x2d)
        # print('pool2d1-',x2d.shape)
        x2d = self.conv2dlayer2a(x2d)
        # x2d = self.conv2dlayer2b(x2d)
        x2d = self.pool2d2(x2d)
        # print('pool2d2-',x2d.shape)

        #x2d = self.conv2dlayer25a_3d(x2d)
        #x2d = self.conv2dlayer25b_3d(x2d)

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
            # z1 = z.mean(3) + z.max(3)[0]
            # print('z1-',z1.shape)
            z = self.Gem(z).squeeze(-1)
            # print('z2-',z2.shape)
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        # print('feature',feature.shape)
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 2, 0).contiguous()
        # print('feature',feature.shape)

        return feature, None

class GL2(nn.Module):
    '''
    global+local(2 block),hidden_dim=128,mean+max()
    '''
    def __init__(self, planes_list=[1, 32, 64, 128], bin_num=32, hidden_dim=128,num_classes=74):
        super(GL2, self).__init__()
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
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
        # x [b ,c, t, h, w]
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, :, 0:1, :, :]), dim=2)
        # print(x.shape)
        # -------------------------------------------------------------
        # x = x.unsqueeze(2).permute(0,2,1,3,4).contiguous()
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

        n, c, h, w = gl.size()
        gl = gl.view(n, c, self.bin_num, -1)
        gl = gl.max(3)[0]+gl.mean(3)
        #gl = self.Gem(gl).squeeze(-1)
        lo = lo.view(n, c, self.bin_num, -1)
        lo = lo.max(3)[0]+lo.mean(3)
        #lo = self.Gem(lo).squeeze(-1)
        feature = torch.cat((gl, lo), 2).permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin[0])  # [62,64,256]
        feature = feature.permute(1, 0, 2).contiguous()  # [64,62,256]
        del x, gl, lo
        return feature,None


    

class tryNet(nn.Module):
    '''一支全局,hidden_num=128'''
    def __init__(self, planes_list=[1, 32, 64, 128, 256], bin_num=32, hidden_dim=128,num_classes=74):
        super(tryNet, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Conv3d(self.planes_list[0], self.planes_list[1], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.layer2 = nn.Conv3d(self.planes_list[1], self.planes_list[1], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        self.layer3 = nn.Conv3d(self.planes_list[1], self.planes_list[2], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.layer4 = nn.Conv3d(self.planes_list[2], self.planes_list[3], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.layer5 = nn.Conv3d(self.planes_list[3], self.planes_list[3], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)

        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.bin_num, self.planes_list[3], self.hidden_dim)))])

        self.Gem = GeM()
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
        # x [b ,c, t, h, w]
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, :, 0:1, :, :]), dim=2)
        # print(x.shape)
        # -------------------------------------------------------------
        # x = x.unsqueeze(2).permute(0,2,1,3,4).contiguous()
        x = F.leaky_relu(self.layer1(x), inplace=True)
        x = F.leaky_relu(self.layer2(x), inplace=True)
        x = F.leaky_relu(self.layer3(x), inplace=True)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.layer4(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.leaky_relu(self.layer5(x), inplace=True)
        x = torch.max(x, 2)[0]  #frame_pooling

        #Feature mapping
        b, c, h , w = x.size()
        f = x.view(b, c, self.bin_num, -1)
        #f = f.mean(3) + f.max(3)[0]
        f = self.Gem(f).squeeze(-1)
        f = f.permute(2, 0, 1).contiguous()
        f = f.matmul(self.fc_bin[0])
        f = f.permute(1, 0, 2).contiguous()   #[b, 16, 256]
        del x
        return f, None

class tryNetlo2v1(nn.Module):
    '''
    局部一支，分2块，hidden_dim=128
    '''
    def __init__(self, planes_list=[1, 32, 64, 128, 256], bin_num=32, hidden_dim=128,num_classes=74):
        super(tryNetlo2v1, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Conv3d(self.planes_list[0], self.planes_list[1], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.layer2 = nn.Conv3d(self.planes_list[1], self.planes_list[1], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        #self.local1 = LOCALBlock(nn.Conv3d(self.planes_list[1], self.planes_list[2], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False), 2)
        #self.local2 = LOCALBlock(nn.Conv3d(self.planes_list[2], self.planes_list[3], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False), 2)
        #self.local3 = LOCALBlock(nn.Conv3d(self.planes_list[3], self.planes_list[3], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False), 2)
        self.local1 = LOCALBlock1(self.planes_list[1], self.planes_list[2],p=2)
        self.local2 = LOCALBlock1(self.planes_list[2], self.planes_list[3],p=2)
        self.local3 = LOCALBlock1(self.planes_list[3], self.planes_list[3],p=2)

        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.bin_num, self.planes_list[3], self.hidden_dim)))])

        self.Gem = GeM()
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
        #x [b ,c, t, h, w]
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, :, 0:1, :, :]), dim=2)
        # print(x.shape)
        #-------------------------------------------------------------
        #x = x.unsqueeze(2).permute(0,2,1,3,4).contiguous()
        x = F.leaky_relu(self.layer1(x), inplace=True)
        x = F.leaky_relu(self.layer2(x), inplace=True)
        x = self.local1(x)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.local2(x)
        x = self.local3(x)
        x = torch.max(x, 2)[0]  #frame_pooling

        #Feature mapping
        b, c, h , w = x.size()
        f = x.view(b, c, self.bin_num, -1)
        #f = f.mean(3) + f.max(3)[0]
        f = self.Gem(f).squeeze(-1)
        f = f.permute(2, 0, 1).contiguous()
        f = f.matmul(self.fc_bin[0])
        f = f.permute(1, 0, 2).contiguous()   #[b, 16, 256]
        del x
        return f,None

class tryNet2(nn.Module):
    '''
    上下两路：全局+局部分2块,hidedeb_dim=128,gem()
    '''
    def __init__(self, planes_list=[1, 32, 64, 128], bin_num=32, hidden_dim=128,num_classes=74):
        super(tryNet2, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Conv3d(self.planes_list[0], self.planes_list[1], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.layer2 = nn.Conv3d(self.planes_list[1], self.planes_list[1], kernel_size=(3, 1, 1), stride=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.global1 = nn.Conv3d(self.planes_list[1], self.planes_list[2], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.global2 = nn.Conv3d(self.planes_list[2], self.planes_list[3], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.global3 = nn.Conv3d(self.planes_list[3], self.planes_list[3], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        self.local1 = LOCALBlock1(self.planes_list[1], self.planes_list[2], p=2)
        self.local2 = LOCALBlock1(self.planes_list[2], self.planes_list[3], p=2)
        self.local3 = LOCALBlock1(self.planes_list[3], self.planes_list[3], p=2)

        self.Gem = GeM()
        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2*self.bin_num, 128, self.hidden_dim)))])
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
        # x [b ,c, t, h, w]
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, :, 0:1, :, :]), dim=2)
        # print(x.shape)
        # -------------------------------------------------------------
        # x = x.unsqueeze(2).permute(0,2,1,3,4).contiguous()
        x = F.leaky_relu(self.layer1(x), inplace=False)
        x = F.leaky_relu(self.layer2(x), inplace=False)

        gl = F.max_pool3d(F.leaky_relu(self.global1(x), inplace=False), kernel_size=(1, 2, 2), stride=(1, 2, 2))
        lo = F.max_pool3d(self.local1(x), kernel_size=(1, 2, 2), stride=(1, 2, 2)) + gl
        gl = F.leaky_relu(self.global2(gl), inplace=False)
        lo = self.local2(lo) + gl
        gl = F.leaky_relu(self.global3(gl), inplace=False)
        lo = self.local3(lo) + gl

        gl = torch.max(gl, 2)[0]  # frame_pooling
        lo = torch.max(lo, 2)[0]  # frame_pooling

        n, c, h, w = gl.size()
        gl = gl.view(n, c, self.bin_num, -1)
        #gl = gl.max(3)[0]+gl.max(3)[0]-gl.mean(3)
        gl = self.Gem(gl).squeeze(-1)
        lo = lo.view(n, c, self.bin_num, -1)
        #lo = lo.max(3)[0]+lo.max(3)[0]-lo.mean(3)
        lo = self.Gem(lo).squeeze(-1)
        feature = torch.cat((gl, lo), 2).permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin[0])  # [62,64,256]
        feature = feature.permute(1, 0, 2).contiguous()  # [64,62,256]
        del x, gl, lo
        return feature,None


    '''
    局部一支，分2块
    '''
    def __init__(self, planes_list=[1, 32, 64, 128, 256], bin_num=32, hidden_dim=128, num_classes=74):
        super(tryNetlo2MS, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Conv3d(self.planes_list[0], self.planes_list[1], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.layer2 = nn.Conv3d(self.planes_list[1], self.planes_list[1], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        #self.local1 = LOCALBlock(nn.Conv3d(self.planes_list[1], self.planes_list[2], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False), 2)
        #self.local2 = LOCALBlock(nn.Conv3d(self.planes_list[2], self.planes_list[3], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False), 2)
        #self.local3 = LOCALBlock(nn.Conv3d(self.planes_list[3], self.planes_list[3], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False), 2)
        self.local1 = LOCALBlock1(self.planes_list[1], self.planes_list[2], P=2)
        self.local2 = LOCALBlock1(self.planes_list[2], self.planes_list[3], P=2)
        self.local3 = LOCALBlock1(self.planes_list[3], self.planes_list[3], P=2)

        self.ms = MultiSpan(self.planes_list[3], self.planes_list[3])
        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.bin_num, 3*self.planes_list[3], self.hidden_dim)))])

        self.Gem = GeM()
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
        # x [b ,c, t, h, w]
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, :, 0:1, :, :]), dim=2)
        # print(x.shape)
        # -------------------------------------------------------------
        # x = x.unsqueeze(2).permute(0,2,1,3,4).contiguous()
        x = F.leaky_relu(self.layer1(x), inplace=True)
        x = F.leaky_relu(self.layer2(x), inplace=True)
        x = self.local1(x)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.local2(x)
        x = self.local3(x)
        x = self.ms(x)
        x = torch.max(x, 2)[0]  #frame_pooling

        #Feature mapping
        b, c, h , w = x.size()
        f = x.view(b, c, self.bin_num, -1)
        #f = f.mean(3) + f.max(3)[0]
        f = self.Gem(f).squeeze(-1)
        f = f.permute(2, 0, 1).contiguous()
        f = f.matmul(self.fc_bin[0])
        f = f.permute(1, 0, 2).contiguous()   #[b, 16, 256]
        del x
        return f, None






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
    model = GaitGL(**kwargs)       #########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return model


if __name__ == "__main__":
    net = c3d_vgg_Fusion(num_classes=74)
    print(params_count(net))
    with torch.no_grad():
        # x = torch.ones(4*3*16*64*44).reshape(4,3,16,64,44)
        x = torch.ones(4 * 1 * 32 * 64 * 44).reshape(4, 1, 32, 64, 44)
        # a = Variable(a.cuda)
        print('x=', x.shape)
        # a,b = net(x)
        # print('a,b=',a.shape,b.shape)
        a,_ = net(x)
        print('a,b=', a.shape)