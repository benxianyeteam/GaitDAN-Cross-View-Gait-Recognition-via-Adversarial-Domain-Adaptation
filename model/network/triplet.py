import torch
import torch.nn as nn
import torch.nn.functional as F



class TripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, feature, label):
        # feature: [n, m, d], label: [n, m]
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).byte().view(-1)  #[h*b*b]
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).byte().view(-1)  #[h*b*b]

        dist = self.batch_dist(feature)   #[h,b,b]
        #print('dist-',dist.shape)
        mean_dist = dist.mean(1).mean(1)  #[h]
        dist = dist.view(-1)  #[h*b*b]
        #print('mean_dist-',mean_dist.shape)
        # hard
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]  #[h,b]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]  #[h,b]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)   #[h,b]

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)  #[h]

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)  #[h,b,8,1]
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)  #[h,b,1,56]
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1) #[h,b*8*56]
        #print('full_hp_dist-', full_hp_dist.shape)
        #print('full_hn_dist-', full_hn_dist.shape)
        #print('full_loss_metric-', full_loss_metric.shape)
        full_loss_metric_sum = full_loss_metric.sum(1)   #[h]
        full_loss_num = (full_loss_metric != 0).sum(1).float()  #[h]

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num  #[h]
        full_loss_metric_mean[full_loss_num == 0] = 0
        #print('full_loss_metric_mean-', full_loss_metric_mean.size())
        #print('hard_loss_metric_mean-', hard_loss_metric_mean.size())
        #print('mean_dist-',mean_dist.size())
        #print('full_loss_num-',full_loss_num.size())
        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):        #[h,b,f]
        x2 = torch.sum(x ** 2, 2)   #[h,b]
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))  #[h,b,b]
        dist = torch.sqrt(F.relu(dist)) #[h,b,b]
        return dist



class GlobalTripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin=0.01):
        super(GlobalTripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, feature, label, pw):
        # feature: [h, b, f], label: [b], pw: [1,1,h]
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(0) == label.unsqueeze(1)).byte().view(-1)  #[b*b]
        hn_mask = (label.unsqueeze(0) != label.unsqueeze(1)).byte().view(-1)  #[b*b]

        dist = self.batch_dist(feature)   #[h,b,b]
        equ_dist = dist.mean(0)           #[b,b]
        pw = pw.mean(0)
        pw = (pw ** 2).squeeze()
        pw = pw.unsqueeze(1).unsqueeze(1)
        #print('pw-size-', pw.shape)
        pw_sum = torch.sum(pw.squeeze())
        #print('pw_sum-size-', pw_sum.shape)
        ada_dist = torch.sum(pw * dist, 0) #[b,b]
        #print('ada_dist-size-',ada_dist.shape)
        ada_dist = ada_dist /pw_sum

        equ_dist = equ_dist.view(-1)  #[h*b*b]
        ada_dist = ada_dist.view(-1)
        #print('equ_dist-size-', equ_dist.shape)
        #print('ada_dist-size-', ada_dist.shape)
        # hard
        '''
        hard_hp_dist = torch.max(torch.masked_select(ada_dist, hp_mask).view(m, -1), 1)[0]  #[b]
        hard_hn_dist = torch.min(torch.masked_select(equ_dist, hn_mask).view(m, -1), 1)[0]  #[b]
        #print('hard_hp_dist-size-', hard_hp_dist.shape)
        #print('hard_hn_dist-size-', hard_hn_dist.shape)
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(-1)   #[b]

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 0)  #[1]
        '''
        # non-zero full
        ada_full_hp_dist = torch.masked_select(ada_dist, hp_mask).view(m, -1, 1)  #[b,8,1]
        ada_full_hn_dist = torch.masked_select(ada_dist, hn_mask).view(m, -1, 1)  #[b,1,56]
        equ_full_hp_dist = torch.masked_select(equ_dist, hp_mask).view(m, -1, 1)  #[b,8,1]
        equ_full_hn_dist = torch.masked_select(equ_dist, hn_mask).view(m, 1, -1)  #[b,1,56]
        #print('ada_full_hp_dist-size-', ada_full_hp_dist.shape)
        #print('ada_full_hn_dist-size-', ada_full_hn_dist.shape)
        #print('equ_full_hp_dist-size-', equ_full_hp_dist.shape)
        #print('equ_full_hn_dist-size-', equ_full_hn_dist.shape)
        full_loss_metric_hp = F.relu(self.margin + ada_full_hp_dist - equ_full_hp_dist).view(-1)
        full_loss_metric_hn = F.relu(self.margin + equ_full_hn_dist - ada_full_hn_dist).view(-1)    #[b*8*56]

        full_loss_metric_sum_hp = full_loss_metric_hp.sum(0)   #[1]
        full_loss_metric_sum_hn = full_loss_metric_hn.sum(0)
        full_loss_num_hp = (full_loss_metric_hp != 0).sum(0).float()  #[1]
        full_loss_num_hn = (full_loss_metric_hn != 0).sum(0).float()  # [1]

        full_loss_metric_mean_hp = full_loss_metric_sum_hp / full_loss_num_hp  #[1]
        full_loss_metric_mean_hp[full_loss_num_hp == 0] = 0
        full_loss_metric_mean_hn = full_loss_metric_sum_hn / full_loss_num_hn  # [1]
        full_loss_metric_mean_hn[full_loss_num_hn == 0] = 0
        #print('full_loss_metric_mean_hp-', full_loss_metric_mean_hp.size())
        #print('full_loss_metric_mean_hn-', full_loss_metric_mean_hn.size())
        full_loss_metric_mean = (full_loss_metric_mean_hp + full_loss_metric_mean_hn)/2
        full_loss_num = full_loss_num_hp + full_loss_num_hn
        return full_loss_metric_mean, full_loss_num

    def batch_dist(self, x):        #[h,b,f]
        x2 = torch.sum(x ** 2, 2)   #[h,b]
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))  #[h,b,b]
        dist = torch.sqrt(F.relu(dist)) #[h,b,b]
        return dist

#####-------------cross-correlation matrix loss-----------------
class CcmaxtrixLoss(nn.Module):
    '''
    输入x维度:[n, c, h]
    '''
    def __init__(self):
        super(CcmaxtrixLoss, self).__init__()
    def forward(self, x):
        n, c, h = x.size()
        x = x.permute(0, 2, 1).contiguous()
        x = F.normalize(x, dim=2)
        x_t = x.permute(0, 2, 1).contiguous()
        cc_m = []
        for i in range(n):
            temp = torch.mm(x[i], x_t[i])
            cc_m.append(temp.unsqueeze(0))
        cc_m = torch.cat(cc_m, 0).sum(0)
        #print('cc_m:{}\n'.format(cc_m))
        diag = torch.diag(cc_m)
        cc_m_diag = torch.diag_embed(diag)
        cc_m = cc_m - cc_m_diag
        cc_m = torch.pow(cc_m, 2)
        full_loss_num = (cc_m != 0).sum().float()
        cc_m = cc_m.sum()/full_loss_num
        #print('cc_m:{}'.format(cc_m))
        return cc_m




