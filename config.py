
conf = {
        "WORK_PATH":r"/home/usr/Diff/open2/work2/",      ###########r"/home/hth/code/HGLnew/work"     /mnt/HDD/dpai1/hth/code/HGLnew/work  /home/gait_group/hth/code/HGLnew/work
        "CUDA_VISIBLE_DEVICES": "6,7,8,9",       ######gai

        "data": {
            'dataset_path': r'/home/usr/dhy/GaitSet-master/dataset/CASIA-B-deletion-1/',    ######gai # r'/home/hth/data/GaitDatasetB_silh_aligned'   r'/mnt/HDD/dpai1/hth/data/GaitDatasetB_silh_aligned'  /home/gait_group/hth/dataset/GaitDatasetB_silh_aligned
            'resolution': '64',
            'dataset': 'CASIA-B',                ######gai CASIA-B
            'pid_num': 74,                  ######gai  74 62 3075
            'pid_shuffle': False,            ######gai
        },
        "model": {
            'hidden_dim': 128,              #######gai
            'lr': 1e-4,                     #######gai
            'hard_or_full_trip': 'full',
            'batch_size': (8, 8),           ######gai
            'restore_iter': 0,              ######gai
            'total_iter': 100000,
            'margin': 0.2,
            'num_workers': 8,
            'frame_num': 30,               ######gai
            #'model_name': 'GMl4STA3glgem1com1GRLT3',           ######gai Globalview7com1  lr1-1.2 lr2-0.7 lr3-1.2
            #'model_name': 'GMl4STA3glgem1com1GRLRE',         ######lr1-2.0 lr2-0.5 lr3-0.8 lr4-1.2
            #'model_name': 'GMl4STA3glgemcomGRLv3',     ######without weight_decracy
            #'model_name': 'GaitGLPid4',            ######lr1-0.5 lr1-2.0
            #'model_name': 'GMl4STA3FPP2ccmcom1',     ###lr-0.01
            #'model_name': 'GMl4STA3FPP2com1lr2',      ###lr1-0.5 lr2-0.7 lr3-1.5
            #'model_name': 'GlobalSTA31FFTlr3',  #lr1-0.5 lr2-0.7  lr3-0.3
            #'model_name': 'GlobalSTA31com1alllr3',
            #'model_name': 'GMl4STA3glgem1com1GRL',
            'model_name': 'GMl4STA3glgem1com1GRL'


        },
    }


'''

conf = {
        "WORK_PATH": r"/mnt/HDD/dpai1/hth/code/HGLnew/work",         ###  /home/gait_group/hth/code/HGLnew/work
        "CUDA_VISIBLE_DEVICES": "3,0,1,2",       ######gai

        "data": {
            'dataset_path': r'/mnt/HDD/dpai1/yaojun/datasets/OUMVLP_cropped',    ######gai # r'/mnt/HDD/dpai1/hth/data/casia-b_ske_selected'  '/home/gait_group/resources/database/OUMVLP_cropped'
            'resolution': '64',
            'dataset': 'OUMVLP',
            'pid_num': 5153,                  ######gai
            'pid_shuffle': False,            ######gai
        },
        "model": {
            'hidden_dim': 256,              #######gai
            'lr': 1e-4,
            'hard_or_full_trip': 'full',
            'batch_size': (20, 8),           ######gai
            'restore_iter': 160000,              ######gai
            'total_iter': 250000,
            'margin': 0.2,
            'num_workers': 16,
            'frame_num': 30,               ######gai
            #'model_name': 'GMl4STA3glgem1com1GRLT3',           ######gai Globalview7com1  lr1-1.2 lr2-0.7 lr3-1.2
            #'model_name': 'GMl4STA3glgem1com1GRLRE',         ######lr1-2.0 lr2-0.5 lr3-0.8 lr4-1.2
            'model_name': 'GMl4STA3glgem1com1GRLbigv1',     ######without weight_decracy
            #'model_name': 'GMl4STA3glgemcom1try1',            ######lr1-0.5 lr1-2.0
            #'model_name': 'GMl4STA3FPP2ccmcom1',     ###lr-0.01
            #'model_name': 'GMl4STA3FPP2com1lr2',      ###lr1-0.5 lr2-0.7 lr3-1.5
            #'model_name': 'GlobalSTA31FFTlr3',  #lr1-0.5 lr2-0.7  lr3-0.3


        },
    }
'''

'''
conf = {
        "WORK_PATH": r"/mnt/HDD/dpai1/hth/code/HGLnew/work",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",       ######gai

        "data": {
            'dataset_path': r'/mnt/HDD/dpai1/hth/data/OUMVLP_cropped', #r'/mnt/HDD/dpai1/hth/data/GaitDatasetB_silh_aligned',   #r'/mnt/HDD/dpai1/hth/data/OUMVLP_cropped'
            'resolution': '64',
            'dataset': 'OUMVLP',
            'pid_num': 5153,                  ######gai
            'pid_shuffle': False,            ######gai
        },
        "model": {
            'hidden_dim': 384,              #######gai
            'lr': 1e-4,
            'hard_or_full_trip': 'full',
            'batch_size': (32, 8),           ######gai
            'restore_iter': 185000,              ######gai
            'total_iter': 250000,
            'margin': 0.2,
            'num_workers': 16,
            'frame_num': 30,               ######gai
            #'model_name': 'GMl4STA3glgem1com1GRLT3',           ######gai Globalview7com1  lr1-1.2 lr2-0.7 lr3-1.2
            'model_name': 'GMl4STA3glgem1com1GRLbigv8',         ######lr1-2.0 lr2-0.5 lr3-0.8 lr4-1.2
            #'model_name': 'GMl4STA3gl1com1partw6smo',
            #'model_name': 'GMl4STA3gl2com1',            ######lr1-0.5 lr1-2.0
            #'model_name': 'GMl4STA3FPP2ccmcom1',     ###lr-0.01
            #'model_name': 'GMl4STA3FPP2com1lr2',      ###lr1-0.5 lr2-0.7 lr3-1.5
            #'model_name': 'GlobalSTA31FFTlr3',  #lr1-0.5 lr2-0.7  lr3-0.3


        },
    }
'''

'''
conf = {
        "WORK_PATH": r"/mnt/HDD/dpai1/hth/code/HGLnew/work",
        "CUDA_VISIBLE_DEVICES": "2,0,1,3,4,5,6,7",       ######gai

        "data": {
            'dataset_path': r'/mnt/HDD/dpai1/hth/data/GaitDatasetB_silh_aligned',    ######gai # r'/mnt/HDD/dpai1/hth/data/GaitDatasetB_silh_aligned' OUMVLP_cropped
            'resolution': '64',
            'dataset': 'CASIA-B',
            'pid_num': 74,                  ######gai
            'pid_shuffle': False,            ######gai
        },
        "model": {
            'hidden_dim': 128,              #######gai
            'lr': 1e-4,
            'hard_or_full_trip': 'full',
            'batch_size': (8, 8),           ######gai
            'restore_iter': 0,              ######gai
            'total_iter': 100000,
            'margin': 0.2,
            'num_workers': 16,
            'frame_num': 30,               ######gai
            #'model_name': 'GMl4STA3glgem1com1GRLT3',           ######gai Globalview7com1  lr1-1.2 lr2-0.7 lr3-1.2
            'model_name': 'GMl4A2glgem1com1GRLtry4',         ######lr1-2.0 lr2-0.5 lr3-0.8 lr4-1.2
            #'model_name': 'GMl4STA3gl1com1partw6smo',
            #'model_name': 'GMl4STA3gl2com1',            ######lr1-0.5 lr1-2.0
            #'model_name': 'GMl4STA3FPP2ccmcom1',     ###lr-0.01
            #'model_name': 'GMl4STA3FPP2com1lr2',      ###lr1-0.5 lr2-0.7 lr3-1.5
            #'model_name': 'GlobalSTA31FFTlr3',  #lr1-0.5 lr2-0.7  lr3-0.3


        },
    }
'''


