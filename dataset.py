from collections import defaultdict
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T


from data_utils import to_sparse_tensor, rand_splits

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit, DGraphFin
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily
from os import path

import networkx as nx
import networkx.algorithms as A
import networkx.algorithms.approximation.maxcut as maxcut
import dgl
import json
import os


def create_homogenous_graph(data):
    row, col = data.edge_index
    graph = dgl.graph((row, col), num_nodes=data.x.shape[0])
    graph.ndata['features'] = data.x
    graph.ndata['labels'] = data.y
    
    return graph
def load_dataset(args, time_bound=[2015,2017],train_idx=0, valid_idx=1, p_ii=1.5, p_ij=0.5):
    if args.dataset == 'twitch':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_twitch_dataset(args.data_dir, train_idx, valid_idx)
    elif args.dataset in 'arxiv' and args.ood_type != "label":
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_arxiv_dataset(args.data_dir, time_bound=time_bound)
    elif args.dataset in ('cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computer', 'coauthor-cs', 'coauthor-physics', 'arxiv'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_graph_dataset(args.data_dir, args.dataset, args.ood_type, p_ii, p_ij)
    else:
        raise ValueError('Invalid dataname')
    
    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_twitch_dataset(data_dir, train_idx, valid_idx):
    transform = T.NormalizeFeatures()
    subgraph_names = ['DE', 'EN', 'ES', 'FR', 'RU']
    dataset_ood_te = []
    for i in range(len(subgraph_names)):
        torch_dataset = Twitch(root=f'{data_dir}Twitch',
                              name=subgraph_names[i], transform=transform)
        dataset = torch_dataset[0]
        dataset.node_idx = torch.arange(dataset.num_nodes)
        if i == train_idx:
            dataset_ind = dataset
        elif i == valid_idx:
            dataset_ood_tr = dataset
        else:
            dataset_ood_te.append(dataset)
    return dataset_ind, dataset_ood_tr, dataset_ood_te


def load_arxiv_dataset(data_dir, time_bound=[2014,2016], inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = ogb_dataset.graph['node_year']

    year_min, year_max = time_bound[0], time_bound[1]
    test_year_bound = [time_bound[1], time_bound[1]+1, time_bound[1]+2, time_bound[1]+3]

    center_node_mask = (year <= year_min).squeeze(1)
    center_node_mask = torch.tensor(center_node_mask)
    if inductive:
        ind_edge_index, _ = subgraph(center_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (year <= year_max).squeeze(1) * (year > year_min).squeeze(1)
    if inductive:
        all_node_mask = (year <= year_max).squeeze(1)
        all_node_mask = torch.tensor(all_node_mask)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in range(len(test_year_bound)-1):
        center_node_mask = (year <= test_year_bound[i+1]).squeeze(1) * (year > test_year_bound[i]).squeeze(1)
        if inductive:
            all_node_mask = (year <= test_year_bound[i+1]).squeeze(1)
            all_node_mask = torch.tensor(all_node_mask)
            
            ood_te_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_te_edge_index = edge_index

        dataset = Data(x=node_feat, edge_index=ood_te_edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)
    return dataset_ind, dataset_ood_tr, dataset_ood_te

def load_proteins_dataset(data_dir, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=f'{data_dir}/ogb')

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    edge_feat = torch.as_tensor(ogb_dataset.graph['edge_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    edge_index_ = to_sparse_tensor(edge_index, edge_feat, ogb_dataset.graph['num_nodes'])
    node_feat = edge_index_.mean(dim=1)

    node_species = torch.as_tensor(ogb_dataset.graph['node_species'])
    species = [0] + node_species.unique().tolist()
    ind_species_min, ind_species_max = species[0], species[3]
    ood_tr_species_min, ood_tr_species_max = species[3], species[5]
    ood_te_species = [species[i] for i in range(5, 8)]

    center_node_mask = (node_species <= ind_species_max).squeeze(1) * (node_species > ind_species_min).squeeze(1)
    if inductive:
        all_node_mask = (node_species <= ind_species_max).squeeze(1)
        ind_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ind_edge_index = edge_index

    dataset_ind = Data(x=node_feat, edge_index=ind_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[center_node_mask]

    center_node_mask = (node_species <= ood_tr_species_max).squeeze(1) * (node_species > ood_tr_species_min).squeeze(1)
    if inductive:
        all_node_mask = (node_species <= ood_tr_species_max).squeeze(1)
        ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
    else:
        ood_tr_edge_index = edge_index

    dataset_ood_tr = Data(x=node_feat, edge_index=ood_tr_edge_index, y=label)
    idx = torch.arange(label.size(0))
    dataset_ood_tr.node_idx = idx[center_node_mask]

    dataset_ood_te = []
    for i in ood_te_species:
        center_node_mask = (node_species == i).squeeze(1)
        dataset = Data(x=node_feat, edge_index=edge_index, y=label)
        idx = torch.arange(label.size(0))
        dataset.node_idx = idx[center_node_mask]
        dataset_ood_te.append(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te



def create_sbm_dataset(data, p_ii=1.5, p_ij=0.5):
    
    n = data.num_nodes

    d = data.edge_index.size(1) / data.num_nodes / (data.num_nodes - 1)
    num_blocks = int(data.y.max()) + 1
    p_ii, p_ij = p_ii * d, p_ij * d
    block_size = n // num_blocks
    block_sizes = [block_size for _ in range(num_blocks-1)] + [block_size + n % block_size]
    edge_probs = torch.ones((num_blocks, num_blocks)) * p_ij
    edge_probs[torch.arange(num_blocks), torch.arange(num_blocks)] = p_ii
    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)

    dataset = Data(x=data.x, edge_index=edge_index, y=data.y)
    dataset.node_idx = torch.arange(dataset.num_nodes)

    return dataset

def create_feat_noise_dataset(data):

    x = data.x
    n = data.num_nodes
    idx = torch.randint(0, n, (n, 2))
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)

    return dataset

def create_dgl_graph(data):
    row, col = data.edge_index
    graph = dgl.graph((row, col), num_nodes=data.x.shape[0])
    graph.ndata['features'] = data.x
    graph.ndata['labels'] = data.y
    
    return graph

def create_label_noise_dataset(data):

    y = data.y
    n = data.num_nodes
    idx = torch.randperm(n)[:int(n * 0.5)]
    y_new = y.clone()
    y_new[idx] = torch.randint(0, y.max(), (int(n * 0.5), ))

    dataset = Data(x=data.x, edge_index=data.edge_index, y=y_new)
    dataset.node_idx = torch.arange(n)

    return dataset


def load_graph_dataset(data_dir, dataname, ood_type, p_ii=1.5, p_ij=0.5):
    transform = T.NormalizeFeatures()
    if dataname in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid', split='public',
                              name=dataname, transform=transform)
        dataset = torch_dataset[0]
        tensor_split_idx = {}
        idx = torch.arange(dataset.num_nodes)
        tensor_split_idx['train'] = idx[dataset.train_mask]
        tensor_split_idx['valid'] = idx[dataset.val_mask]
        tensor_split_idx['test'] = idx[dataset.test_mask]
        dataset.splits = tensor_split_idx
    elif dataname == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Photo', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Computers', transform=transform)
        dataset = torch_dataset[0]
        
    elif dataname == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='CS', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='Physics', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'arxiv': 
        from ogb.nodeproppred import NodePropPredDataset
        ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}ogb')
        edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
        x = torch.as_tensor(ogb_dataset.graph['node_feat'])
        label = torch.as_tensor(ogb_dataset.labels).squeeze(1)
        dataset = Data(x=x, edge_index=edge_index, y=label)
    else:
        raise NotImplementedError

    dataset.node_idx = torch.arange(dataset.num_nodes)
    idx = torch.arange(dataset.num_nodes)
    if ood_type == 'structure':
        dataset_ind = dataset
        dataset_ood_tr = create_sbm_dataset(dataset, p_ii, p_ij)
        dataset_ood_te = create_sbm_dataset(dataset, p_ii, p_ij)
    elif ood_type == 'feature':
        dataset_ind = dataset
        dataset_ood_tr = create_feat_noise_dataset(dataset)
        dataset_ood_te = create_feat_noise_dataset(dataset)
    
    elif ood_type == 'label':
        dataset_ind = dataset
        if dataname == 'cora':
            class_t , ood_te_bar =  4, 4
        elif dataname == 'amazon-photo':
            class_t, ood_te_bar = 5, 5
        elif dataname == 'coauthor-cs':
            class_t, ood_te_bar = 11, 11
        elif dataname == 'coauthor-physics':
            class_t, ood_te_bar = 3, 3
        elif dataname == 'arxiv':
            ind_bar, ood_te_bar = 25, 32
            class_t = ind_bar
        label = dataset.y

        center_node_mask_ind = (label < class_t)
        idx = torch.arange(label.size(0))

        if dataname == 'arxiv':
            center_node_mask_ood_tr = (label >= class_t) * (label < ood_te_bar)
            center_node_mask_ood_te = (label >= ood_te_bar)
        else:
            center_node_mask_ood_tr = (label == class_t)
            center_node_mask_ood_te = (label > class_t)

        edge_index = dataset.edge_index
        center_node_mask = center_node_mask_ind
        center_node_mask = torch.tensor(center_node_mask)
        inductive = True
        label = dataset.y
        
        if inductive:
            ind_edge_index, _ = subgraph(center_node_mask, edge_index)
        else:
            ind_edge_index = dataset.edge_index

        dataset_ind = Data(x=dataset.x, edge_index=ind_edge_index, y=label)
        dataset_ind.node_idx = idx[center_node_mask]
        
        tensor_split_idx = rand_splits(dataset_ind.node_idx)
        dataset_ind.splits = tensor_split_idx
        
        #for OOD tr
        center_node_mask = center_node_mask_ood_tr
    
        if inductive:
            all_node_mask = (label <= ood_te_bar)
            all_node_mask = torch.tensor(all_node_mask)
            ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_tr_edge_index = edge_index

        dataset_ood_tr = Data(x=dataset.x, edge_index=ood_tr_edge_index, y=label)
        dataset_ood_tr.node_idx = idx[center_node_mask]

        dataset_ood_te = []


        center_node_mask = center_node_mask_ood_te
        ood_te_edge_index = edge_index

        dataset_ood_te = Data(x=dataset.x, edge_index=ood_te_edge_index, y=label)
        dataset_ood_te.node_idx = idx[center_node_mask]

    
    elif ood_type == 'density':
        
        graph_dgl = create_dgl_graph(dataset)
        graph_nx = nx.Graph(graph_dgl.to_networkx())
        density = np.array(list(A.clustering(graph_nx).values()))
        print("density is " , np.quantile(density, q=1/3), np.median(density), np.quantile(density, q=2/3))
        density_path=f"{dataname}/density_distribution.json"
        os.makedirs(os.path.dirname(density_path), exist_ok=True)
        json.dump(list(A.clustering(graph_nx).values()), open(density_path, 'w'))
        print("density shape, ", density.shape[0])
        if dataname == 'cora':
            ind_bar, ood_te_bar =0.3, 0.01 #cora
        elif dataname == 'coauthor-cs':
            ind_bar, ood_te_bar = 0.39, 0.18  #coauthor-cs
        elif dataname == 'amazon-photo':
            ind_bar, ood_te_bar = 0.47, 0.28
        elif dataname == 'amazon-computer':
            ind_bar, ood_te_bar = 0.4, 0.23
        elif dataname == 'coauthor-physics':
            ind_bar, ood_te_bar = 0.4, 0.2
        else:
            ind_bar, ood_te_bar = np.quantile(density, q=2/3), np.quantile(density, q=1/3)

        ind_mask = (density>=ind_bar)
        ood_te_mask = (density<ood_te_bar)
        ood_tr_mask = (density < ind_bar) * (density>=ood_te_bar)

        edge_index = dataset.edge_index
        center_node_mask = ind_mask
        center_node_mask = torch.tensor(center_node_mask)
        inductive = True
        label = dataset.y
        
        if inductive:
            ind_edge_index, _ = subgraph(center_node_mask, edge_index)
        else:
            ind_edge_index = dataset.edge_index

        dataset_ind = Data(x=dataset.x, edge_index=ind_edge_index, y=label)
        dataset_ind.node_idx = idx[center_node_mask]
        tensor_split_idx = rand_splits(dataset_ind.node_idx)
        dataset_ind.splits = tensor_split_idx

        center_node_mask = ood_tr_mask
    
        if inductive:
            all_node_mask = (density>=ood_te_bar)
            all_node_mask = torch.tensor(all_node_mask)
            ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_tr_edge_index = edge_index

        dataset_ood_tr = Data(x=dataset.x, edge_index=ood_tr_edge_index, y=label)
        dataset_ood_tr.node_idx = idx[center_node_mask]
        dataset_ood_te = []
        center_node_mask = ood_te_mask
        ood_te_edge_index = edge_index

        dataset_ood_te = Data(x=dataset.x, edge_index=ood_te_edge_index, y=label)
        dataset_ood_te.node_idx = idx[center_node_mask]


    elif ood_type == 'sq_cluster':
        target_path = f"{dataname}/node_sq_cluster_dict.json"
        graph_dgl = create_dgl_graph(dataset)
        graph_nx = nx.Graph(graph_dgl.to_networkx())
        if not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            graph_dgl = create_dgl_graph(dataset)
            graph_nx = nx.Graph(graph_dgl.to_networkx())
            dict_to_save = A.square_clustering(graph_nx)
            json.dump(dict_to_save, open(target_path, "w"))
        else:
            dict_to_save = json.load(open(target_path, "r"))
        density = np.array(list(dict_to_save.values()))
        print("density is " , np.quantile(density, q=1/3), np.median(density), np.quantile(density, q=2/3))
        print("density shape, ", density.shape[0])
        
        if dataname == 'cora':
            ind_bar, ood_te_bar = 0.04, 0.0001 #0.025, 0.001 #cora
        elif dataname == 'coauthor-cs':
            ind_bar, ood_te_bar = 0.06, 0.02 #coauthor-cs
        elif dataname == 'amazon-photo':
            ind_bar, ood_te_bar = 0.14, 0.08 #amazon-photo
        elif dataname == 'amazon-computer':
            ind_bar, ood_te_bar = 0.09, 0.05 #amazon-photo
        elif dataname == 'coauthor-physics':
            ind_bar, ood_te_bar = 0.07, 0.04

        ind_mask = (density>=ind_bar)
        ood_te_mask = (density<ood_te_bar)
        ood_tr_mask = (density < ind_bar) * (density>=ood_te_bar)

        edge_index = dataset.edge_index
        center_node_mask = ind_mask
        center_node_mask = torch.tensor(center_node_mask)
        inductive = True
        label = dataset.y
        
        if inductive:
            ind_edge_index, _ = subgraph(center_node_mask, edge_index)
        else:
            ind_edge_index = dataset.edge_index

        dataset_ind = Data(x=dataset.x, edge_index=ind_edge_index, y=label)
        dataset_ind.node_idx = idx[center_node_mask]
        
        tensor_split_idx = rand_splits(dataset_ind.node_idx)
        dataset_ind.splits = tensor_split_idx

        center_node_mask = ood_tr_mask
    
        if inductive:
            all_node_mask = (density>=ood_te_bar)
            all_node_mask = torch.tensor(all_node_mask)
            ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_tr_edge_index = edge_index

        dataset_ood_tr = Data(x=dataset.x, edge_index=ood_tr_edge_index, y=label)
        dataset_ood_tr.node_idx = idx[center_node_mask]

        dataset_ood_te = []
        center_node_mask = ood_te_mask
        ood_te_edge_index = edge_index
        dataset_ood_te = Data(x=dataset.x, edge_index=ood_te_edge_index, y=label)
        dataset_ood_te.node_idx = idx[center_node_mask]

    elif ood_type == 'max_clique_num':
        target_path = f"{dataname}/node_max_clique_num_dict.json"
        graph_dgl = create_dgl_graph(dataset)
        graph_nx = nx.Graph(graph_dgl.to_networkx())
        if not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            graph_dgl = create_dgl_graph(dataset)
            graph_nx = nx.Graph(graph_dgl.to_networkx())
            dict_to_save = A.number_of_cliques(graph_nx)
            json.dump(dict_to_save, open(target_path, "w"))
        else:
            dict_to_save = json.load(open(target_path, "r"))
        if dataname == 'cora':
            ind_bar, ood_te_bar = 3, 2 #cora
        elif dataname == 'coauthor-cs':
            ind_bar, ood_te_bar = 5, 3 #coauthor-cs
        elif dataname == 'amazon-photo':
            ind_bar, ood_te_bar = 52, 9 #amazon-photo
        elif dataname == 'amazon-computer':
            ind_bar, ood_te_bar = 54, 10 #amazon-photo
        elif dataname == 'coauthor-physics':
            ind_bar, ood_te_bar = 8, 3 #amazon-photo

        max_clique = np.array(list(dict_to_save.values()))
        print("max_clique is " , np.quantile(max_clique, q=1/3), np.median(max_clique), np.quantile(max_clique, q=2/3))
        print("max_clique shape, ", max_clique.shape[0])

        ind_mask = (max_clique>=ind_bar)
        ood_te_mask = (max_clique<ood_te_bar)
        ood_tr_mask = (max_clique < ind_bar) * (max_clique>=ood_te_bar)

        edge_index = dataset.edge_index
        center_node_mask = ind_mask
        center_node_mask = torch.tensor(center_node_mask)
        inductive = True
        label = dataset.y
        
        if inductive:
            ind_edge_index, _ = subgraph(center_node_mask, edge_index)
        else:
            ind_edge_index = dataset.edge_index

        dataset_ind = Data(x=dataset.x, edge_index=ind_edge_index, y=label)
        dataset_ind.node_idx = idx[center_node_mask]
        
        tensor_split_idx = rand_splits(dataset_ind.node_idx)
        dataset_ind.splits = tensor_split_idx

        center_node_mask = ood_tr_mask
    
        if inductive:
            all_node_mask = (max_clique>=ood_te_bar)
            all_node_mask = torch.tensor(all_node_mask)
            ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_tr_edge_index = edge_index

        dataset_ood_tr = Data(x=dataset.x, edge_index=ood_tr_edge_index, y=label)
        dataset_ood_tr.node_idx = idx[center_node_mask]
        dataset_ood_te = []
        center_node_mask = ood_te_mask
        ood_te_edge_index = edge_index

        dataset_ood_te = Data(x=dataset.x, edge_index=ood_te_edge_index, y=label)
        dataset_ood_te.node_idx = idx[center_node_mask]

    elif ood_type == 'trian':
        target_path = f"{dataname}/node_trian.json"
        if not os.path.exists(target_path):
            graph_dgl = create_dgl_graph(dataset)
            graph_nx = nx.Graph(graph_dgl.to_networkx())
            dict_to_save = A.triangles(graph_nx)
            json.dump(dict_to_save, open(target_path, "w"))
        else:
            dict_to_save = json.load(open(target_path, "r"))
        if dataname == 'cora':
            ind_bar, ood_te_bar = 3, 2 #cora
        elif dataname == 'coauthor-cs':
            ind_bar, ood_te_bar = 5, 3 #coauthor-cs
        elif dataname == 'amazon-photo':
            ind_bar, ood_te_bar = 52, 9 #amazon-photo
        max_clique = np.array(list(dict_to_save.values()))
        print("max_clique is " , np.quantile(max_clique, q=1/3), np.median(max_clique), np.quantile(max_clique, q=2/3))
        print("max_clique shape, ", max_clique.shape[0])
        ind_mask = (max_clique>=ind_bar)
        ood_te_mask = (max_clique<ood_te_bar)
        ood_tr_mask = (max_clique < ind_bar) * (max_clique>=ood_te_bar)

        print("node_idx mask:", ind_mask.sum(), ood_tr_mask.sum(), ood_te_mask.sum())

        edge_index = dataset.edge_index
        center_node_mask = ind_mask
        center_node_mask = torch.tensor(center_node_mask)
        inductive = True
        label = dataset.y
        
        if inductive:
            ind_edge_index, _ = subgraph(center_node_mask, edge_index)
        else:
            ind_edge_index = dataset.edge_index

        dataset_ind = Data(x=dataset.x, edge_index=ind_edge_index, y=label)
        dataset_ind.node_idx = idx[center_node_mask]
        
        tensor_split_idx = rand_splits(dataset_ind.node_idx)
        dataset_ind.splits = tensor_split_idx

        center_node_mask = ood_tr_mask
        if inductive:
            all_node_mask = (max_clique>=ood_te_bar)
            all_node_mask = torch.tensor(all_node_mask)
            ood_tr_edge_index, _ = subgraph(all_node_mask, edge_index)
        else:
            ood_tr_edge_index = edge_index

        dataset_ood_tr = Data(x=dataset.x, edge_index=ood_tr_edge_index, y=label)
        dataset_ood_tr.node_idx = idx[center_node_mask]
        dataset_ood_te = []
        center_node_mask = ood_te_mask
        ood_te_edge_index = edge_index
        dataset_ood_te = Data(x=dataset.x, edge_index=ood_te_edge_index, y=label)
        dataset_ood_te.node_idx = idx[center_node_mask]

    else:
        raise NotImplementedError
    return dataset_ind, dataset_ood_tr, dataset_ood_te


