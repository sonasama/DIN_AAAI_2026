import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_evaluation_results, criteria
from Dataloader import load_data
# from model import MvGAGNN
import copy
import random
from args import parameter_parser
args = parameter_parser()
from model import Net1
# from model import MultipleOptimizer


def train(args, device):
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    feature_list, adj_hat, labels, idx_labeled, idx_unlabeled = load_data(args, device=device)
    print('Labeled sample:', len(idx_labeled))
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    input_dims = []
    num_views = len(feature_list)
    for i in range(num_views):
        input_dims.append(feature_list[i].shape[1])

    if args.model_type == '2' or args.model_type == '3':
        model = Net1(input_dims, args.hidden_dims, num_views, num_classes, args.r_view, args.r_feat, args.num_layers, args.dropout,
                 args.type, device).to(device)
    else:
        model = Net1(input_dims, args.hidden_dims, num_views,feature_list[0].shape[0],
                 num_classes, args.r_view, args.r_feat, args.num_layers, args.dropout,
                 args.type, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    begin_time = time.time()
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    train_acc_list = []
    train_f1_list = []
    loss_list = []
    val_acc_list = []
    val_f1_list = []

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            z = model(feature_list, adj_hat)
            optimizer.zero_grad()

            loss = F.cross_entropy(z[idx_labeled], labels[idx_labeled])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                z = model(feature_list, adj_hat)

                pred_labels1 = torch.argmax(z, 1).cpu().detach().numpy()

                val_acc1, val_f11 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled],
                                                           pred_labels1[idx_unlabeled])
                val_acc_list.append(val_acc1)
                val_f1_list.append(val_f11)

                pbar.set_postfix({'Loss': '{:.6f}'.format(loss.item()),

                                  'val_acc': '{:.2f}'.format(val_acc1 * 100),
                                  'val_f1': '{:.2f}'.format(val_f11 * 100),
                                  'best_val_acc': '{:.2f}'.format(best_acc * 100),
                                  'best_val_f1': '{:.2f}'.format(best_f1 * 100)}
                                 )
                pbar.update(1)

    cost_time = time.time() - begin_time
    model.eval()
    z = model(feature_list, adj_hat)
    pred_labels = torch.argmax(z, 1).cpu().detach().numpy()

    acc, f1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
    print("Evaluating the model")
    print("ACC: {:.2f}, F1: {:.2f}".format(acc * 100, f1 * 100))

    return acc * 100, f1 * 100, cost_time, loss_list, val_acc_list, val_f1_list, train_acc_list, train_f1_list
