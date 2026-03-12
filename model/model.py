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
from .network import TripletLoss, SetNet
from .network import HGL_tri_cse
from .utils import TripletSampler
from glob import glob
from os.path import join

import cv2
from sklearn.utils import shuffle


class Model:
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

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        print('num_works:{}'.format(self.num_workers))
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.img_size = img_size

        self.dataset = save_name.split('_')[1]
        self.logPath = os.path.join(r'/home/gait_group/hth/code/HGLnew/log/{}'.format(self.dataset), 'log_{}_{}_{}_{}_{}'.format(model_name, train_pid_num,self.P,self.M, frame_num)+'.txt')
        '''
        self.logPath = os.path.join(r'/mnt/HDD/dpai1/hth/code/HGLnew/log/{}'.format(self.dataset),
                                    'log_{}_{}_{}_{}_{}'.format(model_name, train_pid_num, self.P, self.M,
                                                                frame_num) + '.txt')
        '''
        self.log_f = open(self.logPath, 'a')


        print(train_pid_num)
        self.log_f.write('train_pid_num:{}'.format(train_pid_num)+'\n')
        self.m_resnet = HGL_tri_cse.c3d_vgg_Fusion(num_classes=self.train_pid_num)   #################gai
        self.m_resnet = self.m_resnet.cuda()

        print(torch.cuda.device_count(),batch_size)
        self.log_f.write('cuda.device_count:{}'.format(torch.cuda.device_count()) + '\n')
        self.m_resnet = nn.DataParallel(self.m_resnet, device_ids=[0,2,3])        ####################gai
        print('model_parameters_num:{}'.format(sum(p.numel() for p in self.m_resnet.parameters())), file=self.log_f)
        print(self.m_resnet, file=self.log_f)
        print(self.m_resnet)


        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        #self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.triplet_loss.cuda()

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        #self.hard_loss_metric_fft = []
        #self.full_loss_metric_fft = []
        #self.full_loss_num_fft = []
        #self.dist_list_fft = []
        self.mean_dist = 0.01

        self.optimizer = optim.Adam([
            {'params': self.m_resnet.parameters()}], lr=self.lr)

        print(self.optimizer)


        # self.losscse= []

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
                if len(frame_set)>self.frame_num-1:
                    choiceframe_set = frame_set[:len(frame_set)-self.frame_num+1]
                    frame_id_list= random.choices(choiceframe_set, k=1)
                    for i in range(self.frame_num-1):  
                        frame_id_list.append(frame_id_list[0]+(i+1))
                    _ = [feature.loc[frame_id_list].values for feature in sample]
                else:
                    frame_id_list = [0]
                    for i in range(len(frame_set)-1):  
                        frame_id_list.append(frame_id_list[0]+(i+1))    
                    len_frame_id_list = len(frame_id_list)
                    for ll in range(self.frame_num-len_frame_id_list):
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
                frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                frame_id_list = sorted(frame_id_list)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs1 = []
        count = 1
        for sq in range(len(seqs)):
            seqs1.append(Order_select_frame(sq))
            count +=1
        seqs = seqs1

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

        self.m_resnet.train()
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
        print('-len(train_loader)-',len(train_loader))
        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()
        print(train_label_set)
        train_view_set = list(self.train_source.view_set)  ####view
        train_view_set.sort()  ####view
        print(train_view_set)
        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            # self.exp_lr_scheduler.step()
            self.restore_iter += 1
            #print(self.restore_iter)
            self.optimizer.zero_grad()
            labelint = []
            if self.train_pid_num==5153:
                for li in label:
                    labelint.append((int(li)-1)//2)              
            else:
                for li in label:
                    labelint.append(int(li)-1)
            # print(labelint)
            #---------------targets-----------------------
            targets = np.array(labelint)
            #---------------seq-----------------------
            seq=np.array(seq)
            seq = np.float32(seq)
            seq = seq.squeeze(0)

            targets = torch.from_numpy(targets)
            targets = targets.cuda()
            seq = torch.from_numpy(seq)
            #seq = seq.squeeze(0)
            seq = seq.cuda()


            targets = Variable(targets)
            seq = Variable(seq)
            triplet_feature, _ = self.m_resnet(seq)   #####

            #--------------tri-------------------------
            triplet_feature = triplet_feature.permute(1, 0, 2).contiguous()     #######zhuyi
            triplet_label = targets.unsqueeze(0)
            triplet_label = triplet_label.repeat(triplet_feature.size(0), 1)
            #print(triplet_feature.size())
            #print(triplet_label[0,:])
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)

            # --------------tri-------------------------
            #fft_feature = fft_feature.permute(1, 0, 2).contiguous()  #######zhuyi
            #(full_loss_metric_fft, hard_loss_metric_fft, mean_dist_fft, full_loss_num_ftt
            # ) = self.triplet_loss(fft_feature, triplet_label)


            if self.hard_or_full_trip == 'hard':
                # loss = hard_loss_metric.mean()
                print('-----hard-----')
            elif self.hard_or_full_trip == 'full':
                lossspa = full_loss_metric.mean()
                #lossfft = full_loss_metric_fft.mean()
                loss = lossspa

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            #self.hard_loss_metric_fft.append(hard_loss_metric_fft.mean().data.cpu().numpy())
            #self.full_loss_metric_fft.append(full_loss_metric_fft.mean().data.cpu().numpy())
            #self.full_loss_num_fft.append(full_loss_num_ftt.mean().data.cpu().numpy())
            #self.dist_list_fft.append(mean_dist_fft.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()
                self.save()


            if self.restore_iter % 100000 == 0:
                print('----------------LR-------------------')
                self.optimizer = optim.Adam([
                    {'params': self.m_resnet.parameters()}], lr=0.00001)
                '''xin+'''
            if self.restore_iter % 225001 == 0:
                print('----------------LR-------------------')
                self.optimizer = optim.Adam([
                    {'params': self.m_resnet.parameters()}], lr=0.2*0.00001)

            if self.restore_iter % 100 == 0:
                #self.save()
                print('iter {}:'.format(self.restore_iter), end='', file=self.log_f)
                print(', h_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='', file=self.log_f)
                print(', f_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='', file=self.log_f)
                print(', f_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='', file=self.log_f)
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='', file=self.log_f)
                '''
                print(', h_loss_metric_fft={0:.8f}'.format(np.mean(self.hard_loss_metric_fft)), end='', file=self.log_f)
                print(', f_loss_metric_fft={0:.8f}'.format(np.mean(self.full_loss_metric_fft)), end='', file=self.log_f)
                print(', f_loss_num_fft={0:.8f}'.format(np.mean(self.full_loss_num_fft)), end='', file=self.log_f)
                self.mean_dist_fft = np.mean(self.dist_list_fft)
                print(', mean_dist_fft={0:.8f}'.format(self.mean_dist_fft), end='', file=self.log_f)
                '''
                # print(', cse_loss_num={0:.8f}'.format(np.mean(self.losscse)), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='', file=self.log_f)
                print(', hard or full=%r' % self.hard_or_full_trip, file=self.log_f)


                print('iter {}:'.format(self.restore_iter), end='')
                print(', h_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', f_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', f_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                # print(', cse_loss_num={0:.8f}'.format(np.mean(self.losscse)), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []



            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
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

            counttt=0
            for i, x in enumerate(data_loader):
                seq, view, seq_type, label, batch_frame = x
                seq=np.array(seq)
                seq = np.float32(seq)
                seq = torch.from_numpy(seq)
                seq = seq.squeeze(0)
                #seq = seq.unsqueeze(1)
                seq = Variable(seq.cuda())
                #batcht,channelt,framet,ht,wt = seq.size()
                outputs, _ = self.m_resnet(seq)
                # outputs = self.m_resnet(seq)
                if counttt%1000==0:
                    print(label, seq_type,view,'--',outputs.shape)
                counttt +=1
                '''enhance'''
                '''
                n, num_bin, _ = outputs.size()
                outputs=outputs.permute(1,0, 2).contiguous()
                feature_list.append(outputs.data.cpu().numpy())
                del outputs
                '''
                n,_,_ = outputs.size()
                # n,_ = outputs.size()
                outputs = outputs.view(n,-1)
                outputs = torch.mean(outputs,dim=0)
                outputs = outputs.unsqueeze(0)
                n,_ = outputs.size()
                # print(outputs.view(n, -1).shape)
                outputs = outputs.view(n, -1).data.cpu().numpy()
                feature_list.append(outputs)


                view_list += view
                seq_type_list += seq_type
                label_list += label

        #return np.concatenate(feature_list, 1), view_list, seq_type_list, label_list   #enhance
        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

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
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))


class Model_1:
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
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
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

        self.encoder = SetNet(self.hidden_dim).float()
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'
        self.train_writer1 = tensorboardX.SummaryWriter(os.path.join('logs', self.model_name, self.save_name))

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
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

        batch[0] = seqs
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, label_prob = self.encoder(*seq, batch_frame)    #feature[64,62,256]


            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()  #[62,64,256]
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1) #[62,64]
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())


            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()
                self.save()

            if self.restore_iter % 100 == 0:
                #self.save()
                self.train_writer1.add_scalar('loss', loss, self.restore_iter)
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            '''
            if self.restore_iter % 100 == 0:
                m = initialization(conf, False)[0]

                # load model checkpoint of iteration opt.iter
                print('Loading the model of iteration %d...' % self.restore_iter)
                # m.load(opt.iter)
                m.load(self.restore_iter)
                print('Transforming...')
                time = datetime.now()
                # test = m.transform('test', opt.batch_size)
                test = m.transform('test', 1)
                print('Evaluating...')
                acc = evaluation(test, conf['data'])
                print('Evaluation complete. Cost:', datetime.now() - time)
                print('Acc:', np.mean(acc[0, :, :, 0]))

                save_name = '_'.join([self.model_name, str(self.restore_iter), str(np.mean(acc[0, :, :, 0]))])
                np.save('./result/' + save_name, acc)
                data = pd.DataFrame(acc[0, :, :, 0])
                writer = pd.ExcelWriter('./result/' + save_name + '_Excel.xlsx')
                data.to_excel(writer, 'page_1', float_format='%.5f')
                writer.close()
            '''
            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            # print(batch_frame, np.sum(batch_frame))

            feature, _ = self.encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))

