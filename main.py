import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from logger import Logger_classify, Logger_detect, save_result
from data_utils import evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, rand_splits
from dataset import load_dataset
from parse import parser_add_main_args
from baselines import *
import time

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# twitch
subgraph_names = ['DE', 'EN', 'ES', 'FR', 'RU']

# arxiv
time_bound_1 = [2012, 2013, 2014, 2015, 2016]
time_bound_2 = [2014, 2015, 2016, 2017]

# else
p_ij_bound = list((np.arange(3, 7)/10.0)) 
p_ii_bound = list((np.arange(12, 17)/10.0)) 

if args.test_all:
    if args.dataset == 'twitch':
        ix_list = list(range(len(subgraph_names)))
        jx_list = list(range(len(subgraph_names)))
    elif args.dataset == 'arxiv':
        ix_list = time_bound_1
        jx_list = time_bound_2
    else:
        ix_list = p_ij_bound
        jx_list = p_ii_bound

else:
    if args.dataset == 'twitch':
        ix_list = [0]
        jx_list = [1]
    elif args.dataset == 'arxiv':
        ix_list = [2015]
        jx_list = [2017]
    else:
        ix_list = [0.5]
        jx_list = [1.5]

for ix in ix_list:
    for jx in jx_list:
        if ix == jx:
            continue
        ### Load and preprocess data ###
        if args.dataset == 'twitch':
            dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args, train_idx=ix, valid_idx=jx)
        elif args.dataset == 'arxiv':
            dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args, time_bound=[ix,jx])  
        else:
            dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args, p_ii=jx, p_ij=ix)

        if len(dataset_ind.y.shape) == 1:
            dataset_ind.y = dataset_ind.y.unsqueeze(1)
        if len(dataset_ood_tr.y.shape) == 1:
            dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
        if isinstance(dataset_ood_te, list):
            for data in dataset_ood_te:
                if len(data.y.shape) == 1:
                    data.y = data.y.unsqueeze(1)
        else:
            if len(dataset_ood_te.y.shape) == 1:
                dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            print("load fixed splits")
            pass

        else:
            dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

        c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
        d = dataset_ind.x.shape[1]

        print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
            + f"classes {c} | feats {d}")
        print(f"ood tr dataset {args.dataset}: all nodes {dataset_ood_tr.num_nodes} | centered nodes {dataset_ood_tr.node_idx.shape[0]} | edges {dataset_ood_tr.edge_index.size(1)}")
        if isinstance(dataset_ood_te, list):
            for i, data in enumerate(dataset_ood_te):
                print(f"ood te dataset {i} {args.dataset}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
        else:
            print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")


        if args.method == 'msp':
            model = MSP(d, c, args).to(device)
        elif args.method == "knn":
            model = KNN(d, c, args).to(device)
        elif args.method == 'OE':
            model = OE(d, c, args).to(device)
        elif args.method == "ODIN":
            model = ODIN(d, c, args).to(device)
        elif args.method == "Mahalanobis":
            model = Mahalanobis(d, c, args).to(device)
        elif args.method == 'bc':
            model = TopoOOD(d, c, args).to(device)
        if args.dataset in ('proteins', 'ppi'):
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.NLLLoss()

        if args.dataset in ('proteins', 'ppi', 'twitch'):
            eval_func = eval_rocauc
        else:
            eval_func = eval_acc

        if args.mode == 'classify':
            logger = Logger_classify(args.runs, args)
        else:
            logger = Logger_detect(args.runs, args)
        

        model.train()
        ood_threshold = []
        best_epoch = None
        best_valid_loss = float('inf')
        best_model = None
        for run in range(args.runs):
            model.reset_parameters()
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_val = float('-inf')
            for epoch in range(args.epochs):
                current_time = time.time()
                model.train()
                optimizer.zero_grad()
                loss, cluster_centers = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
                loss.backward()
                optimizer.step()
                training_time = time.time() - current_time
                
                if args.mode == 'classify':
                    result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
                    logger.add_result(run, result)

                    if epoch % args.display_step == 0:
                        print(f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * result[0]:.2f}%, '
                            f'Valid: {100 * result[1]:.2f}%, '
                            f'Test: {100 * result[2]:.2f}%')
                else:
                    current_time = time.time()
                    result, test_ind_score, test_ood_score = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device, cluster_centers, return_score=True,k=args.knn)
                    inf_time = time.time() - current_time
                    valid_loss = result[-1].item()
                    logger.add_result(run, result)
                    if epoch % args.display_step == 0:
                        print(f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'AUROC: {100 * result[0]:.2f}%, '
                            f'AUPR: {100 * result[1]:.2f}%, '
                            f'FPR95: {100 * result[2]:.2f}%, '
                            f'Test Score: {100 * result[-2]:.2f}%')
                
            logger.print_statistics(run)

        results = logger.print_statistics()

        if args.dataset == 'twitch':
            save_result(results, args, suffix=f'{subgraph_names[ix]}-{subgraph_names[jx]}')
        else:
            save_result(results, args, suffix=f'{ix}-{jx}')