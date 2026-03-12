from datetime import datetime
import numpy as np
import argparse
import os
from model.initialization import initialization
from model.utils import evaluation
from model.utils import evaluation_en, evaluation_en_global,evaluation_en_without
from config import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result

def de_diag_oulp(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 3.0
    if not each_angle:
        result = np.mean(result)
    return result


# Exclude identical-view cases
def de_diag13(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 13.0
    if not each_angle:
        result = np.mean(result)
    return result


if __name__ == '__main__':
    iterall = []
    for i in range(50, 40,5):
        iterall.append(i * 1000)
    for iter in iterall:
        parser = argparse.ArgumentParser(description='Test')

        parser.add_argument('--batch_size', default='1', type=int,
                            help='batch_size: batch size for parallel test. Default: 1')
        parser.add_argument('--cache', default=False, type=boolean_string,
                            help='cache: if set as TRUE all the test data will be loaded at once'
                                 ' before the transforming start. Default: FALSE')
        opt = parser.parse_args()

        # Exclude identical-view cases

        m = initialization(conf, test=opt.cache)[0]

        # load model checkpoint of iteration opt.iter
        print('Loading the model of iteration %d...' % iter)
        m.load(iter)
        #smo_iter = 10                       ######注意注意
        #pw = m.loadpw(iter,smo_iter)         ######注意注意
        print('Transforming...')
        time = datetime.now()
        test = m.transform('test', opt.batch_size)
        #test = m.transform('test', pw, opt.batch_size)  #######注意注意
        print(len(test), opt.batch_size)
        print('Evaluating...')
        #acc = evaluation_en_global(test, conf['data'], pw)     #######注意注意
        acc = evaluation_en(test, conf['data'])
        print('Evaluation complete. Cost:', datetime.now() - time)

        #--------------new-----------------
        '''
        if dataset == 'OUMVLP':
            acc_w = evaluation_en_without(test, conf['data'])
            print('Evaluation complete. Cost:', datetime.now() - time)
        '''
        #----------------------------------

        dataset = conf['data']['dataset']
        pid_num = conf['data']['pid_num']
        model_name = conf['model']['model_name']

        if dataset == 'CASIA-B':
            # Print rank-1 accuracy of the best model
            # e.g.
            # ===Rank-1 (Include identical-view cases)===
            # NM: 95.405,     BG: 88.284,     CL: 72.041
            '''
            for i in range(1):
                print('===Rank-%d (Include identical-view cases)===' % (i + 1))
                print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                    np.mean(acc[0, :, :, i]),
                    np.mean(acc[1, :, :, i]),
                    np.mean(acc[2, :, :, i])))
            '''

            # Print rank-1 accuracy of the best model,excluding identical view cases
            # e.g.
            # -------Rank-1 (Exclude identical-view cases)-------
            # NM: 94.964,     BG: 87.239,     CL: 70.355


            # Print rank-1 accuracy of the best model (Each Angle)
            # e.g.
            # ----Rank-1 of each angle (Exclude identical-view cases)---
            # NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
            # BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
            # CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]


            np.set_printoptions(precision=2, floatmode='fixed')
            for i in range(1):
                print('---Rank-%d of each angle (Exclude identical-view cases)---' % (i + 1))
                print('NM:', de_diag(acc[0, :, :, i], True))
                print('BG:', de_diag(acc[1, :, :, i], True))
                print('CL:', de_diag(acc[2, :, :, i], True))

            nm = round(de_diag(acc[0, :, :, 0]), 3)
            bg = round(de_diag(acc[1, :, :, 0]), 3)
            cl = round(de_diag(acc[2, :, :, 0]), 3)
            mean = round((nm + bg + cl) / 3, 3)
            #save_name = '_'.join([model_name, str(pid_num), 'en_', 'smo_1'.format(smo_iter), str(iter), str(nm), str(bg), str(cl), str(mean)])  ######注意注意
            save_name = '_'.join(
                [model_name, str(pid_num), 'en_1', str(iter), str(nm), str(bg), str(cl),
                 str(mean)])
            directory=r'/home/usr/Diff/open2/DHY_new/DHY/result/{}/{}/'.format(dataset, model_name) + save_name
            if not os.path.exists(directory):  
                os.makedirs(directory)       
            np.save(directory, acc)
            #np.save(r'/home/gait_group/hth/code/HGLnew/result/{}/{}/'.format(dataset, model_name) + save_name, acc)
            for i in range(1):
                print('---Rank-%d (Exclude identical view cases)---' % (i + 1))
                print('NM: %.3f,\tBG: %.3f,\tCL: %.3f, \tMean: %.3f' % (
                    de_diag(acc[0, :, :, i]),
                    de_diag(acc[1, :, :, i]),
                    de_diag(acc[2, :, :, i]),
                    mean))

        elif dataset == 'OULP':
            acc_exclude = round(de_diag_oulp(acc[0, :, :, 0]), 3)
            save_name = '_'.join([model_name, str(pid_num), 'en_', str(iter), str(acc_exclude)])
            # writer1.add_scalar('acc_exclude', acc_exclude, int(iter))
            np.save(r'/mnt/HDD/dpai1/hth/code/HGLnew/result/{}/{}/'.format(dataset, model_name) + save_name, acc)
            for i in range(1):
                print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
                print('acc_exclude:%.3f' % (de_diag_oulp((acc[0, :, :, i]))))
            print(acc[0, :, :, 0])


        else:
            print(acc.shape)
            for i in range(1):
                print('===Rank-%d (Include identical-view cases)===' % (i + 1))
                print('00: %.3f' % (
                    np.mean(acc[0, :, :, i])))

            for i in range(1):
                print('---Rank-%d (Exclude identical view cases)---' % (i + 1))
                print('00: %.3f' % (
                    de_diag13(acc[0, :, :, i])))
            np.set_printoptions(precision=2, floatmode='fixed')
            for i in range(1):
                print('---Rank-%d of each angle (Exclude identical-view cases)---' % (i + 1))
                print('00:', de_diag13(acc[0, :, :, i], True))

            vd = round(de_diag13(acc[0, :, :, 0]), 3)
            save_name = '_'.join([model_name, str(pid_num), str(iter), 'en_', str(vd)])
            np.save(r'/mnt/HDD/dpai1/hth/code/HGLnew/result/{}/{}/'.format(dataset, model_name) + save_name, acc)
            # print('NM:', acc[0,:,:,0])
            # print('BG:', acc[1,:,:,0])
            # print('CL:', acc[2,:,:,0])
            # print(acc[:,:,:,0].shape)

