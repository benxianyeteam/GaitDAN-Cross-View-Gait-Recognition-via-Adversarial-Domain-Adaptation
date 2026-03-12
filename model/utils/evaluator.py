import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def cuda_dist_en(x, y):
    #x = torch.from_numpy(x).cuda()
    #y = torch.from_numpy(y).cuda()
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(
        2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2))
    dist = torch.sqrt(F.relu(dist)).mean(0)
    return dist


def cuda_dist_en_global(x, y, pw):
    #x = torch.from_numpy(x).cuda()
    #y = torch.from_numpy(y).cuda()
    x = torch.from_numpy(x)     #[h,b,f]
    y = torch.from_numpy(y)     #[h,b,f]
    dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(
        2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2))
    dist = torch.sqrt(F.relu(dist))
    pw = pw.mean(0)
    pw = (pw ** 2).squeeze()
    pw = pw.unsqueeze(1).unsqueeze(1)
    # print('pw-size-', pw.shape)
    pw_sum = torch.sum(pw.squeeze())
    # print('pw_sum-size-', pw_sum.shape)
    ada_dist = torch.sum(pw * dist, 0)  # [b,b]
    # print('ada_dist-size-',ada_dist.shape)
    ada_dist = ada_dist / pw_sum

    return ada_dist


def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['vd-00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['vd-01']]}

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    #gallery_x = feature[:, gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    #probe_x = feature[:, pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc


def evaluation_en(data, config):
    dataset = config['dataset'].split('-')[0]
    #feature, view, seq_type, label = data
    feature, view, seq_type, label, batch_frame= data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['vd-00']],
                      'OULP':[['seq01']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['vd-01']],
                        'OULP':[['seq00']]}

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    #gallery_x = feature[gseq_mask, :]
                    gallery_x = feature[:, gseq_mask, :]   #[h,b,f]
                    gallery_y = label[gseq_mask]           #[b]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    #pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view]) & np.isin(label, gallery_y)   ###去掉不全的
                    #probe_x = feature[pseq_mask, :]
                    probe_x = feature[:, pseq_mask, :]     #[h,b,f]
                    probe_y = label[pseq_mask]             #[b]

                    dist = cuda_dist_en(probe_x, gallery_x)
                    #idx = dist.sort(1)[1].cpu().numpy()
                    idx = dist.sort(1)[1].numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc


def evaluation_en_without(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['vd-00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['vd-01']]}

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    #gallery_x = feature[gseq_mask, :]
                    gallery_x = feature[:, gseq_mask, :]   #[h,b,f]
                    gallery_y = label[gseq_mask]           #[b]

                    #pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view]) & np.isin(label, gallery_y)   ###去掉不全的
                    #probe_x = feature[pseq_mask, :]
                    probe_x = feature[:, pseq_mask, :]     #[h,b,f]
                    probe_y = label[pseq_mask]             #[b]

                    dist = cuda_dist_en(probe_x, gallery_x)
                    #idx = dist.sort(1)[1].cpu().numpy()
                    idx = dist.sort(1)[1].numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc


def evaluation_en_global(data, config, pw):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['vd-00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['vd-01']]}

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    #gallery_x = feature[gseq_mask, :]
                    gallery_x = feature[:, gseq_mask, :]   #[h,b,f]
                    gallery_y = label[gseq_mask]           #[b]

                    #pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view]) & np.isin(label, gallery_y)   ###去掉不全的
                    #probe_x = feature[pseq_mask, :]
                    probe_x = feature[:, pseq_mask, :]     #[h,b,f]
                    probe_y = label[pseq_mask]             #[b]

                    dist = cuda_dist_en_global(probe_x, gallery_x, pw)
                    #idx = dist.sort(1)[1].cpu().numpy()
                    idx = dist.sort(1)[1].numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc