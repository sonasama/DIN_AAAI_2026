import os
import warnings
import random
import torch
import configparser
import numpy as np
from args import parameter_parser
from utils import tab_printer
from train import train
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cuda')

    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)

    all_ACC = []
    all_F1 = []
    all_TIME = []


    for i in range(args.n_repeated):
        ACC, F1, Time, loss, acc, f1, train_acc, train_f1 = train(args, device)

        all_ACC.append(ACC)
        all_F1.append(F1)
        all_TIME.append(Time)


    print("====================")
    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC), np.std(all_ACC)))
    print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1), np.std(all_F1)))
    print("====================")

    with open("./results" + '/{}.txt'.format(args.dataset), 'a', encoding='utf-8') as f:
        f.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
                'dataset:{}'.format(args.dataset) + '\n'
                'epochs:{} | lr:{} | wd:{} | dropout:{} | hidden:{} | layers:{}| r_view:{} | r_feat:{} | knns:{} | ratio:{} | type:{} | Ablation:{}'.format(args.num_epoch,
                args.lr, args.weight_decay, args.dropout,args.hidden_dims,args.num_layers, args.r_view, args.r_feat, args.knns, args.ratio, args.type, args.model_type) + '\n'
                'ACC_mean:{:.4f} | ACC_std:{:.4f}'.format(np.mean(all_ACC), np.std(all_ACC)) + '\n'
                'F1_mean:{:.4f} | F1_std:{:.4f}'.format(np.mean(all_F1), np.std(all_F1)) + '\n'
                'RunTime:{:.4f}s'.format(np.mean(all_TIME)) + '\n'
                '----------------------------------------------------------------------------' + '\n')

