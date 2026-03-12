import math
import os
import os.path as osp
import random
import sys
from datetime import datetime
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
from torch.optim import lr_scheduler
from .network import TripletLoss, CcmaxtrixLoss, GlobalTripletLoss
from .network import HGL_tri_cse  ##############
#from .network import HGL_tri_cse_selfattn ############
from .utils import TripletSampler
from glob import glob
from os.path import join
import cv2
from sklearn.utils import shuffle
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


def regressloss(predict, groundtruth):
    l = nn.MSELoss()
    return l(predict, groundtruth)


class Model_tri_cse_grl:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=112):


        seed = 0  # seed-0/seed1-5/seed2-10/seed3-20/seed4-40
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        # self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.img_size = img_size

        self.dataset = save_name.split('_')[1]


        self.logPath = os.path.join(r'/home/usr/Diff/open2/DHY_new/DHY/work/log/{}'.format(self.dataset),
                                    'log_{}_{}_{}_{}_{}'.format(model_name, train_pid_num, self.P, self.M,
                                                                frame_num) + '.txt')
        '''
        self.logPath = os.path.join(r'/home/gait_group/hth/code/HGLnew/log/{}'.format(self.dataset),
                                    'log_{}_{}_{}_{}_{}'.format(model_name, train_pid_num, self.P, self.M,
                                                                frame_num) + '.txt')
        '''
        self.log_f = open(self.logPath, 'a')

        print(train_pid_num)
        self.log_f.write('train_pid_num:{}'.format(train_pid_num) + '\n')
        self.m_resnet = HGL_tri_cse.c3d_vgg_Fusion(num_classes=self.train_pid_num)  ###############
        self.m_resnet = self.m_resnet.cuda()

        print(torch.cuda.device_count(), batch_size)
        self.log_f.write('cuda.device_count:{}'.format(torch.cuda.device_count()) + '\n')
        self.m_resnet = nn.DataParallel(self.m_resnet)
        print('model_parameters_num:{}'.format(sum(p.numel() for p in self.m_resnet.parameters())), file=self.log_f)
        print(self.m_resnet, file=self.log_f)
        print(self.m_resnet)

        print('-------------------- CrossEntropyLoss--------------------------------')
        # self.criterion = nn.CrossEntropyLoss().cuda()
        self.criterion = LabelSmoothingCrossEntropy().cuda()
        self.criterion_view = nn.CrossEntropyLoss().cuda()
        # self.critfft = LabelSmoothingCrossEntropy().cuda()
        # self.crit_all = nn.CrossEntropyLoss().cuda()

        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.triplet_loss.cuda()

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.weight_decay = 5e-4         ########改
        #self.weight_decay = 0  ########改

        self.optimizer = optim.Adam([{'params': self.m_resnet.parameters()}], lr=self.lr,
                                    weight_decay=self.weight_decay)
        # self.optimizer = optim.Adam([{'params': self.m_resnet.module.attn_param}], lr=self.lr, weight_decay=self.weight_decay)
        # self.optimizer = optim.Adam([
        #     {'params': self.m_resnet.parameters()}], lr=self.lr)

        print(self.optimizer)

        self.losscse = []
        self.lossview = []
        self.losstotal = []
        self.lossccmax = []  ########新增

        self.sample_type = 'all'

    def collate_fn(self, batch):
        # print('collate_fn-1-',batch.shape)
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def Order_select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                if len(frame_set) > self.frame_num - 1:
                    choiceframe_set = frame_set[:len(frame_set) - self.frame_num + 1]
                    frame_id_list = random.choices(choiceframe_set, k=1)
                    for i in range(self.frame_num - 1):
                        frame_id_list.append(frame_id_list[0] + (i + 1))
                    _ = [feature.loc[frame_id_list].values for feature in sample]
                else:
                    frame_id_list = [0]
                    for i in range(len(frame_set) - 1):
                        frame_id_list.append(frame_id_list[0] + (i + 1))
                    len_frame_id_list = len(frame_id_list)
                    for ll in range(self.frame_num - len_frame_id_list):
                        frame_id_list.append(frame_id_list[ll])
                    _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        def select_frame(index):
            sample = seqs[index]
            # print(sample[0].shape)
            frame_set = frame_sets[index]

            if self.sample_type == 'random':
                # frame_id_list = random.choices(frame_set, k=self.frame_num)
                frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                frame_id_list = sorted(frame_id_list)
                # print('-1-',frame_id_list)
                # print('-2-',sorted(frame_id_list))
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        # seqs = list(map(select_frame, range(len(seqs))))
        # seqs = list(map(LMS_select_frame, range(len(seqs))))
        seqs1 = []
        count = 1
        for sq in range(len(seqs)):
            # print('--------------------',count)
            seqs1.append(Order_select_frame(sq))
            # seqs1.append(select_frame(sq))
            count += 1
        seqs = seqs1
        # print(len(seqs),len(seqs[0]),len(seqs[0][0]))
        # print()
        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            # print('--2--')
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                len(frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]
            # print(gpu_num,batch_per_gpu,batch_frames)
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
            seqs = [np.asarray([
                np.pad(seqs[j][_],
                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                       'constant',
                       constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

            # seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        batch[0] = seqs
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)
        # m_resnet = generate_resnet18()
        # print(self.m_resnet)
        self.m_resnet.train()
        # self.m_resnet.eval()
        # print(self.m_resnet.module.attn_param)
        # self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        # print('---1--')
        # print('batch_size-',self.batch_size)
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=self.num_workers)
        print('-len(train_loader)-', len(train_loader))
        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()
        train_view_set = list(self.train_source.view_set)   ####view
        train_view_set.sort()
        print(train_view_set)
                               ####view
        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            # self.exp_lr_scheduler.step()
            self.restore_iter += 1
            self.optimizer.zero_grad()
            print(view)
            labelint = []
            if self.train_pid_num == 5153:
                for li in label:
                    labelint.append((int(li) - 1) // 2)
            else:
                for li in label:
                    labelint.append(int(li) - 1)
            # print(labelint)
            # ---------------targets-----------------------
            targets = np.array(labelint)
            targets_view = [train_view_set.index(l) for l in view]  ##view
            print(targets_view)
            targets_view = np.array(targets_view)                   ##view
            # ---------------seq-----------------------
            seq = np.array(seq)
            seq = np.float32(seq)
            seq = seq.squeeze(0)
            # print(targets.shape,seq.shape)
            # --------------shuffle-----------------
            #seq, targets = shuffle(seq, targets)
            seq, targets, targets_view = shuffle(seq, targets, targets_view)
            # --------------------------------------

            targets = torch.from_numpy(targets)
            targets = targets.cuda()
            targets_view = torch.from_numpy(targets_view)   ##view
            targets_view = targets_view.cuda()              ##view
            seq = torch.from_numpy(seq)

            # seq = torch.stack((seq,seq,seq),dim=1)    ###########改
            seq = seq.cuda()
            # print('--',seq.shape)

            targets = Variable(targets)
            targets_view = Variable(targets_view)  ##view
            seq = Variable(seq)
            # print('-input-',seq.shape,targets.shape)

            triplet_feature, outputscse, feaview,x = self.m_resnet(seq)  # [b,128,64] [b,64,74]###########!!!!!!
            # print('pw-size-', pw.shape)

            # --------------ccmatrix--------------------
            # ccm_feature = triplet_feature

            # --------------tri-------------------------
            triplet_feature = triplet_feature.permute(2, 0, 1).contiguous()
            # print('-triplet_feature-',triplet_feature.shape)
            global_triplet_label = targets
            triplet_label = targets.unsqueeze(0)
            triplet_label = triplet_label.repeat(triplet_feature.size(0), 1)
            # print('-triplet_label-',triplet_label.shape)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num) = self.triplet_loss(triplet_feature,
                                                                                               triplet_label)
            # ------------------cse----------------------
            n, hpp, classn = outputscse.size()
            outputscse = outputscse.view(n * hpp, -1)
            # targetsall = targets                                ####!!!!!
            targets = targets.unsqueeze(1)
            targets = targets.repeat(1, hpp)
            # targets = targets.view(n, -1).squeeze(1)
            targets = targets.view(n * hpp, -1).squeeze(1)
            # print('-targets-',targets.shape)

            # -----------------view------------------------
            n, hpp, classn = feaview.size()
            feaview = feaview.view(n * hpp, -1)
            targets_viewall = targets_view  ##viewall
            targets_view = targets_view.unsqueeze(1)
            targets_view = targets_view.repeat(1, hpp)
            # targets_view = targets_view.view(n, -1).squeeze(1)
            targets_view = targets_view.view(n * hpp, -1).squeeze(1)
            # print('-targets_view-',targets_view.shape)

            # ------------------fft----------------------
            # n, hpp, classn = outputsfft.size()
            # outputsfft = outputsfft.view(n*hpp, -1)

            if self.hard_or_full_trip == 'hard':
                # loss = hard_loss_metric.mean()
                print('-----hard-----')
            elif self.hard_or_full_trip == 'full':
                cse = self.criterion(outputscse, targets)
                cse_view = self.criterion_view(feaview, targets_view)
                # fft = self.critfft(outputsfft, targets)
                losstri = full_loss_metric.mean()
                # global_losstri = global_full_loss_metric
                loss = cse + losstri + cse_view
                '''
                if self.restore_iter < 20000:
                    loss = cse + losstri + global_losstri
                else:
                    loss = cse + losstri + 10*global_losstri
                '''

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())
            self.losscse.append(cse.data.cpu().numpy())
            self.lossview.append(cse_view.data.cpu().numpy())
            # self.lossfft.append(fft.data.cpu().numpy())  ####!!!!
            # self.lossccmax.append(ccm.data.cpu().numpy())
            # self.global_full_loss_metric.append(global_full_loss_metric.data.cpu().numpy())  ####!!!!
            # self.global_full_loss_num.append(global_full_loss_num.data.cpu().numpy())        ####!!!!
            self.losstotal.append(loss.data.cpu().numpy())


            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()



            if self.restore_iter % 5000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()
                self.save()

            if self.restore_iter % 185001 == 0:
                print('----------------LR-------------------')
                print('---weiht_decay:{}-----'.format(self.weight_decay))
                self.optimizer = optim.Adam([
                    {'params': self.m_resnet.parameters()}], lr=0.00001, weight_decay=self.weight_decay)

            if self.restore_iter % 200000 == 0:
                print('----------------LR-------------------')
                self.weight_decay = 1*5e-4
                self.optimizer = optim.Adam([
                    {'params': self.m_resnet.parameters()}], lr=0.5*0.00001, weight_decay=self.weight_decay)

            if self.restore_iter % 100 == 0:
                print('iter {}:'.format(self.restore_iter), end='', file=self.log_f)
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='', file=self.log_f)
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='', file=self.log_f)
                # print(', g_full_loss_metric={0:.8f}'.format(np.mean(self.global_full_loss_metric)), end='', file=self.log_f)   ####!!!!!
                # print(', g_full_loss_num={0:.8f}'.format(np.mean(self.global_full_loss_num)), end='', file=self.log_f)         ####!!!!!
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='', file=self.log_f)
                print(', cse_loss={0:.8f}'.format(np.mean(self.losscse)), end='', file=self.log_f)
                print(', view_loss={0:.8f}'.format(np.mean(self.lossview)), end='', file=self.log_f)
                # print(', ccm_loss={0:.8f}'.format(np.mean(self.lossccmax)), end='', file=self.log_f)
                # print(', fft_loss={0:.8f}'.format(np.mean(self.lossfft)), end='', file=self.log_f)  ####!!!!!
                print(',cse+tri={0:.8f}'.format(np.mean(self.losstotal)), end='', file=self.log_f)  ####!!!!!
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='', file=self.log_f)
                print(', h or f=%r' % self.hard_or_full_trip, file=self.log_f)

                print('iter {}:'.format(self.restore_iter), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                # print(', g_full_loss_metric={0:.8f}'.format(np.mean(self.global_full_loss_metric)), end='')  ####!!!!!
                # print(', g_full_loss_num={0:.8f}'.format(np.mean(self.global_full_loss_num)), end='')  ####!!!!!
                # self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', cse_loss={0:.8f}'.format(np.mean(self.losscse)), end='')
                print(', view_loss={0:.8f}'.format(np.mean(self.lossview)), end='')
                # print(', fft_loss={0:.8f}'.format(np.mean(self.lossfft)), end='')    ####!!!!!
                # print(', ccm_loss={0:.8f}'.format(np.mean(self.lossccmax)), end='')
                print(',cse+tri={0:.8f}'.format(np.mean(self.losstotal)), end='')  ####!!!!!
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', h or f=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                # self.global_full_loss_metric = []  ####!!!!!
                # self.global_full_loss_num = []     ####!!!!!
                self.full_loss_num = []
                self.dist_list = []

                self.losscse = []
                self.lossview = []   ####view
                self.losstotal = []

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, pw, batch_size=1):
        with torch.no_grad():
            self.m_resnet.eval()
            source = self.test_source if flag == 'test' else self.train_source
            self.sample_type = 'all'
            data_loader = tordata.DataLoader(
                dataset=source,
                batch_size=batch_size,
                sampler=tordata.sampler.SequentialSampler(source),
                collate_fn=self.collate_fn,
                pin_memory=True,
                num_workers=self.num_workers)

            feature_list = list()
            view_list = list()
            seq_type_list = list()
            label_list = list()
            sample_list = list()

            counttt = 0
            for i, x in enumerate(data_loader):
                seq, view, seq_type, label, batch_frame = x

                seq = np.array(seq)
                '''gai'''
                '''
                n,t,h,w = seq.size()
                if t>30:
                    sample = seq[:,0:30,:,:]
                else:
                    sample = seq.repeat(1,30,1,1)
                    sample = sample[:,0:30,:,:]
                print(sample.size())
                sample_list +=  sample.reshape(1,-1)
                '''
                seq = np.float32(seq)
                seq = torch.from_numpy(seq)
                seq = seq.squeeze(0)

                seqtemp = seq
                # seq = torch.stack((seq,seq,seq),dim=1)   #gai

                seq = Variable(seq.cuda())

                # outputs,_,outputsall = self.m_resnet(seq)
                # _, outputs = self.m_resnet(seq)  ####!!!!!
                outputs, outputspred, _ , sample= self.m_resnet(seq)  ####!!!!!
                # if counttt % 10 == 0:
                # print('-------------------------------------------------')
                # print(outputspred.max(2))

                if counttt % 10000 == 0:
                    print('-------new---')
                    print(label, seq_type, view, '--', outputs.shape)
                counttt += 1

                '''enhance'''

                n, num_bin, _ = outputs.size()
                outputs = outputs.permute(2, 0, 1).contiguous()
                feature_list.append(outputs.data.cpu().numpy())

                view_list += view
                seq_type_list += seq_type
                label_list += label
                sample_list.append(sample.data.cpu().numpy())   ###可视化


                '''
                n,_,_ = outputs.size()                 #[b,128,64]
                outputs = outputs.view(n,-1)           #[b,128*64]
                outputs = torch.mean(outputs,dim=0)    #[128*64]
                outputs = outputs.unsqueeze(0)         #[1,128*64]
                n,_ = outputs.size()
                # print(outputs.view(n, -1).shape)
                outputs = outputs.view(n, -1).data.cpu().numpy() #[1,128*64]
                feature_list.append(outputs)

                view_list += view
                seq_type_list += seq_type
                label_list += label
                '''

        # return np.concatenate(feature_list, 1), view_list, seq_type_list, label_list  # enhance
        # return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
        return np.concatenate(feature_list, 1), view_list, seq_type_list, label_list, np.concatenate(sample_list,0)  # enhance + 可视化

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.m_resnet.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        print(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter)))
        self.m_resnet.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        # print('--------no optimizer---------------------------')
        self.optimizer.load_state_dict(torch.load(osp.join(
             'checkpoint', self.model_name,
             '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))

    def savepw(self, partweight, smo_iter):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        np.save(osp.join('checkpoint', self.model_name, '{}-{}-{:0>5}-partweight'.format(
            self.save_name, smo_iter, self.restore_iter)), partweight)

    def loadpw(self, restore_iter, smo_iter):
        path = osp.join(
            'checkpoint', self.model_name,
            '{}-{}-{:0>5}-partweight.npy'.format(self.save_name, smo_iter, restore_iter))
        print(path)
        pw = np.load(path, allow_pickle=True)
        pw = torch.from_numpy(np.mean(pw, 0)).unsqueeze(0).unsqueeze(0)
        print(pw)
        # self.m_resnet.module.partweight = nn.parameter(torch.from_numpy(np.mean(pw,0)).unsqueeze(0).unsqueeze(0))
        '''
        for name, param in self.m_resnet.named_parameters():
            #print(name, ';', param.size())
            if name == 'module.partweight':
               print('param_raw:{}'.format(param))
               self.m_resnet.partweight = nn.parameter(torch.from_numpy(pw).unsqueeze(0).unsqueeze(0))
            #         for name, param in self.m_resnet.named_parameters():
            #             if name == 'module.partweight':
            #                #param = torch.from_numpy(np.mean(pw,0)).unsqueeze(0).unsqueeze(0).cuda()
            #                print('param_new:{}'.format(param))
            #                print('----------------------------------')
            #             # print(name, '', param.size())
        '''
        return pw