import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import json
from torch_geometric.utils import degree
from backbone import *
from kmeans_pytorch import kmeans
import math
from torch.autograd import Variable
import torch.autograd as autograd
import networkx as nx
import dgl

class KNN(nn.Module):
    def __init__(self, d, c, args):
        super(KNN, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn,
                               use_aux=True)  
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=args.use_bn)   
    
    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def get_embedding(self, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        aux_logits = self.encoder.get_aux_output(x, edge_index)
        neg_energy = aux_logits
        if args.use_prop: 
            neg_energy = self.propagation(neg_energy, edge_index, args.K, args.alpha)
        return neg_energy[node_idx].cpu()
    
    def cal_dirichlet(self, dataset, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        # logits = self.encoder.intermediate_forward(x, edge_index)#torch.Size([9498, 2])
        logits = self.encoder(x, edge_index)
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()#torch.Size([9498])
        base = torch.sqrt(1+d)

        for i in range(logits.shape[1]):
            logits[:, i] = torch.div(logits[:, i], base)

        t1 = logits[col, :]
        t2 = logits[row, :] #torch.Size([315774, 2])
        value_row = torch.ones_like(row) * 1. / d[row] 

        value_row = torch.nan_to_num(value_row, nan=0.0, posinf=0.0, neginf=0.0) 
        value_col = torch.ones_like(col) * 1. / d[col]
        value_col = torch.nan_to_num(value_col, nan=0.0, posinf=0.0, neginf=0.0)

        #sqrt(a*b)
        # value = torch.mul(value_col, value_row).sqrt() #torch.Size([315774]) 
        #mean
        value = (value_row + value_col) / 2
        e = torch.sum((t1 - t2).pow(2), dim=1).sqrt()#torch.Size([315774]) 

        e = torch.sum(torch.mul(value, e)) * 0.5 / value.shape[0]
        # print("e is, ", e)
        return e
    
    def cal_dirichlet_node(self, dataset, device, node_idx, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)# edge index shape is:  torch.Size([2, 315774])
        # logits = self.encoder.intermediate_forward(x, edge_index)# logits shape,  torch.Size([9498, 2])
        logits = self.encoder(x, edge_index)
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float() #torch.Size([9498])
        base = torch.sqrt(1+d)
        for i in range(logits.shape[1]):
            logits[:, i] = torch.div(logits[:, i], base)
        t1 = logits[col, :] #torch.Size([315774, 2])
        t2 = logits[row, :]
        per_e = torch.sum((t1 - t2).pow(2), dim=1)#.sqrt()  #torch.Size([315774]) 

        value_row = torch.ones_like(row) * 1. / d[row] 
        value_row = torch.nan_to_num(value_row, nan=0.0, posinf=0.0, neginf=0.0) 
        value_col = torch.ones_like(col) * 1. / d[col]
        value_col = torch.nan_to_num(value_col, nan=0.0, posinf=0.0, neginf=0.0)
        #sqrt(a*b)
        # value = torch.mul(value_col, value_row).sqrt() 
        #mean
        value = (value_row + value_col) / 2
        # for i in value:
        #     if i==0:
        #         print("we have zero in value")
        p = torch.mul(value, per_e) #torch.Size([315774])
        e = torch.zeros(N).to(device)
        e = e.index_add(0, row, p)
        # return e[node_idx]
        return e
    
    def detect(self, train_set, train_idx, test_set, node_idx, device, args): # dataset, node_idx, device, args):
        '''return negative energy, a vector for all input nodes'''
        ftrain = self.get_embedding(train_set, train_idx, device, args).cpu().numpy()
        id_train_size, feat_dim = ftrain.shape
        ftest = self.get_embedding(test_set, node_idx, device, args).cpu().numpy()
        rand_ind = np.random.choice(id_train_size, id_train_size, replace=False)

        # print('feat_dim', feat_dim)
        index = faiss.IndexFlatL2(feat_dim)
        # print("index shape", index)
        index.add(ftrain[rand_ind])

        D, _ = index.search(ftest, args.knn, )
        scores_in = -D[:,-1]
        # print("k", k)
        # print('ftrain.shape: ', ftrain.shape)
        # print('ftest.shape: ', ftest.shape)
        # print('D.shape: ', D.shape)
        # print("_", _.shape)
        # print(scores_in.shape)
        # print(scores_in[:10])

        return torch.tensor(scores_in)


    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        if len(e.shape) == 1:
            e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        # print('e.shape', e.shape)
        if e.shape[1] == 1:
            return e.squeeze(1)
        else:
            return e

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)

        # get predicted logits from gnn classifier
        logits_in = self.encoder(x_in, edge_index_in)
        logits_out = self.encoder(x_out, edge_index_out)
        aux_logits_in = self.encoder.get_aux_output(x_in, edge_index_in)
        aux_logits_out = self.encoder.get_aux_output(x_out, edge_index_out)

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        # compute supervised training loss
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        if args.use_reg: # if use energy regularization
            if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
                logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
                logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
                energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
                energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
            else: # for single-label multi-class classification
                energy_in = aux_logits_in
                energy_out = aux_logits_out
                # print("shape detect: ", energy_in.shape, energy_out.shape)
            if args.use_prop: # use energy belief propagation
                energy_in = self.propagation(energy_in, edge_index_in, args.K, args.alpha)[train_in_idx]
                energy_out = self.propagation(energy_out, edge_index_out, args.K, args.alpha)[train_ood_idx]
                # print("shape detect: ", energy_in.shape, energy_out.shape)
            else:
                energy_in = energy_in[train_in_idx]
                energy_out = energy_out[train_ood_idx]

            # truncate to have the same length
            if energy_in.shape[0] != energy_out.shape[0]:
                min_n = min(energy_in.shape[0], energy_out.shape[0])
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]

            pred_in = F.log_softmax(energy_in, dim=1)
            pred_out = F.log_softmax(energy_out, dim=1)
            
            deng_in = self.cal_dirichlet_node(dataset_ind, device, train_in_idx, args)[train_in_idx]
            deng_out = self.cal_dirichlet_node(dataset_ood, device, train_ood_idx, args)[train_ood_idx]
            # print("deng shape are: ", deng_in.shape, deng_out.shape)
            
            if deng_in.shape[0] != deng_out.shape[0]:
                min_n = min(deng_in.shape[0], deng_out.shape[0])
                deng_in = deng_in[:min_n]
                deng_out = deng_out[:min_n]

            reg_loss_2 = torch.mean(F.relu(deng_in - args.m_in) ** 2 + F.relu(args.m_out - deng_out) ** 2)
            in_loss = criterion(pred_in, torch.tensor([1 for _ in range(pred_in.shape[0])]).to(device))
            ood_loss = criterion(pred_out, torch.tensor([0 for _ in range(pred_out.shape[0])]).to(device))
            reg_loss = in_loss + ood_loss

            loss = sup_loss + args.lamda * reg_loss + args.lamda2 * reg_loss_2
            loss = args.lamda2 * reg_loss_2
        else:
            loss = sup_loss
        

        return loss, torch.zeros((1, 1))


class TopoOOD(nn.Module):
    def __init__(self, d, c, args):
        self.cur_epoch = 0
        self.cur_detect = 0
        self.cur_epoch_median = 0
        self.embedding_epoch = 0
        super(TopoOOD, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn,
                               use_aux=True)  
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=args.use_bn)   
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout) 
        elif args.backbone == 'jknet':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout) 
        elif args.backbone == 'mlp':
            self.encoder = MLP(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout) 

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def get_embedding(self, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        aux_logits = self.encoder.get_aux_output(x, edge_index)

        neg_energy = aux_logits

        if args.use_prop: 
            neg_energy = self.propagation(neg_energy, edge_index, args.K, args.alpha)
            
        return neg_energy[node_idx].cpu()


    def cal_dirichlet_node(self, dataset, device, node_idx, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        N = x.shape[0]
        row, col = edge_index
        d = degree(row, N).float()#torch.Size([9498])
        base = torch.sqrt(1+d)
        for i in range(logits.shape[1]):
            logits[:, i] = torch.div(logits[:, i], base)
        t1 = logits[col, :] #torch.Size([315774, 2])
        t2 = logits[row, :]
        per_e = torch.sum((t1 - t2)**2, dim=1) #torch.Size([315774]) 
        value_row = torch.ones_like(row) * 1. / d[row] 
        value_row = torch.nan_to_num(value_row, nan=0.0, posinf=0.0, neginf=0.0) 

        p = torch.mul(value_row, per_e) #torch.Size([315774])     
        e = torch.zeros(N).to(device)
        e = e.index_add(0, row, p)
        return e
    
    def detect(self, test_set, node_idx, device, args, k=100, t=(None, None, None)): # dataset, node_idx, device, args):
        '''return negative energy, a vector for all input nodes'''
        # result = pred_in[:,1] + 100 * self.cal_dirichlet(test_set, node_idx, device, args)
        d_eng = self.cal_dirichlet_node(test_set, device, node_idx, args)
        # print("deng, ", float(torch.mean(d_eng[node_idx])), float(torch.max(d_eng[node_idx])), float(torch.median(d_eng[node_idx])))
        cal_embedding = False
        t1, t2, t3 = t
        if cal_embedding:
            k1, k2, k3 = self.cal_embedding_center(test_set, device, t1, t2,t3, args)
            print("from ind center", k1)
            print("from ood center", k2)
            print("from all center", k3)
        if args.use_d_prop:
            d_eng = self.d_propagation(d_eng, test_set.edge_index, device, args.K_d, args.alpha2) 
        
        result = args.deng_level * d_eng[node_idx]
        
        return result.cpu()
    
    def cal_embedding_center(self, test_set, device, center_in, center_out_tr, center_all, args):
        x_test, edge_index_test = test_set.x.to(device), test_set.edge_index.to(device)
        logits_test = self.encoder(x_test, edge_index_test)
        center_test = torch.mean(logits_test, dim=0)
        # crop three tensor to the same size
        dist_from_in = torch.mean((center_in - center_test)**2)
        dist_from_ood = torch.mean((center_out_tr - center_test)**2)
        # print(torch.mean(torch.vstack((center_in,center_out_tr)),dim=0).shape)
        dist_from_all = torch.mean((center_all - center_test)**2)
        return float(dist_from_in), float(dist_from_ood), float(dist_from_all)
    
    def d_propagation(self, e, edge_index, device, prop_layers=1, alpha2=0.5):
        N = e.shape[0]
        row, col = edge_index
        # d = degree(col, N).float()
        d = degree(row, N).float()
        # d_norm = 1. / d[col]
        d_norm = 1. / d[row]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        e_previous = e
        e_next = e

        for _ in range(prop_layers):        
            p = torch.mul(value.to(device), e_previous[col.to(device)].to(device)) #torch.Size([315774])
            e_next = torch.zeros(N).to(device)
            e_next = e_next.index_add(0, row.to(device), p)
            e_next = e_next * alpha2 + e_previous * (1 - alpha2)
            e_previous = e_next
        return e_next
    
    def get_n_hop_neighbors(G, node, n):
        """
        Get n-hop neighbors of a given node.
        Parameters:
        - G: A NetworkX graph.
        - node: The node for which n-hop neighbors are required.
        - n: The number of hops.
        Returns:
        - A list of n-hop neighbors of the given node.
        """
        length = nx.single_source_shortest_path_length(G, source=node, cutoff=n)
        return [node for node, distance in length.items() if distance <= n]  # Notice <= instead of ==
    
    def draw_graph_weight(self, dataset, edge_index, device, prop_layers=1, alpha2=0.5):
        center_node =[5]
        
        graph = dgl.graph((row, col), num_nodes=dataset.x.shape[0])
        graph.ndata['features'] = dataset.x
        graph.ndata['labels'] = dataset.y
        graph_original = nx.DiGraph(graph.to_networkx())
        
        n_hop_neigbors = self.get_n_hop_neighbors(graph_original, 5, 5)
        subG = graph_original.subgraph(n_hop_neigbors)
        node_values = {node: -1 for node in subG.nodes()}
        adj_matrix = nx.adjacency_matrix(subG)
        adj_array = adj_matrix.toarray()
        # Retrieve the row and col edge indices
        row, col = np.where(adj_array > 0)
        # List of edges represented by row and column indices
        edges = list(zip(row, col))
        per_e = torch.ones(row.shape) #torch.Size([315774]) 
        value_row = torch.ones_like(row) * 1. / d[row] 

        value_row = torch.nan_to_num(value_row, nan=0.0, posinf=0.0, neginf=0.0) 

        p = torch.mul(value_row, per_e) #torch.Size([315774])     
        e = torch.zeros(N).to(device)
        e = e.index_add(0, row, p)
        N = e.shape[0]
        row, col = edge_index
        # d = degree(col, N).float()
        d = degree(row, N).float()
        # d_norm = 1. / d[col]
        subgraph_list = center_node
        graph_original.nodes[center_node[0]]['value'] = 1
        for i in range(prop_layers):
            for j in center_node:
                neighbors_of_node_1 = list(graph_original.neighbors(j))
                subgraph_list.append(neighbors_of_node_1)
                graph_original.nodes[j]['value'] = 1 
        
        graph_original.nodes[neighbors_of_node_1]['value'] = graph_original.degree[center_node]
        d_norm = 1. / d[row]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


        
        e_previous = e
        e_next = e
        for _ in range(prop_layers):        
            p = torch.mul(value.to(device), e_previous[col.to(device)].to(device)) #torch.Size([315774])
            e_next = torch.zeros(N).to(device)
            e_next = e_next.index_add(0, row.to(device), p)
            e_next = e_next * alpha2 + e_previous * (1 - alpha2)
            e_previous = e_next
        return e_next
    
    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        if len(e.shape) == 1:
            e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        # print('e.shape', e.shape)
        if e.shape[1] == 1:
            return e.squeeze(1)
        else:
            return e

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        # print(dataset_ind)
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)
        y_in = dataset_ind.y.to(device)

        logits_in = self.encoder(x_in, edge_index_in)
        logits_out = self.encoder(x_out, edge_index_out)
        logits_all = torch.vstack((logits_in, logits_out))

        # logits_in.shape[0]
        # print("logits_all shape", logits_in.shape, logits_out.shape, logits_all.shape)

        center_in = torch.mean(logits_in, dim=0)
        center_out = torch.mean(logits_out,dim=0)
        center_all = torch.mean(logits_all,dim=0)

        # print("center_in", torch.sum(center_in**2))
        # print("center_out", torch.sum(center_out**2))
        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx
        # print("loss compute: ", y_in[train_in_idx].shape, logits_out[train_ood_idx].shape)

        plot_embedding = False
        if plot_embedding:

            logits_all_att = torch.vstack((logits_in[train_in_idx], logits_out[train_ood_idx])).cpu().detach().numpy()
            y_out_label = int(logits_out[train_ood_idx].shape[1])
            temp = int(logits_out[train_ood_idx].shape[0])
            # print(y_out_label, temp)
            y_out = torch.ones((temp, 1)).to(device) * y_out_label
            # print("y_out,",y_out_label, y_out.shape )
            label_all_att = torch.vstack((y_in[train_in_idx], y_out)).cpu().detach().numpy()
            print("label_all_att shape", label_all_att.shape)
            # np.save(f"./embedding/{args.dataset}/embedding_in_{args.ood_type}_{self.embedding_epoch}.npy", logits_in_att)
            # np.save(f"./embedding/{args.dataset}/embedding_out_{args.ood_type}_{self.embedding_epoch}.npy", logits_out_att)
            np.save(f"./embedding/{args.dataset}/{args.dataset}_embedding_{args.ood_type}_{self.embedding_epoch}.npy", logits_all_att)
            np.save(f"./embedding/{args.dataset}/{args.dataset}_label_{args.ood_type}_{self.embedding_epoch}.npy", label_all_att)
            self.embedding_epoch += 1


        # compute supervised training loss
        pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
        sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        if args.use_reg: 
            deng_in = self.cal_dirichlet_node(dataset_ind, device, train_in_idx, args)
            deng_out = self.cal_dirichlet_node(dataset_ood, device, train_ood_idx, args)

            if args.use_d_prop:
                deng_in = self.d_propagation(deng_in, edge_index_in, device, args.K_d, args.alpha2)[train_in_idx]
                deng_out = self.d_propagation(deng_out, edge_index_out, device, args.K_d, args.alpha2)[train_ood_idx]
            else:
                deng_in = deng_in[train_in_idx]
                deng_out = deng_out[train_ood_idx]

            reg_loss_2 = torch.mean(F.relu(deng_in - args.m_in) ** 2) + torch.mean(F.relu(args.m_out - deng_out) ** 2)
            loss = sup_loss +  args.lamda2 * reg_loss_2
        else:
            loss = sup_loss
        return loss, (center_in, center_out, center_all)



class Kmeans(nn.Module):
    def __init__(self, d, c, args):
        super(Kmeans, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn,
                               use_aux=True)  
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=args.use_bn)   
    
    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)


    def detect(self, train_set, train_idx, test_set, node_idx, device,cluster_centers ,args, k=100): # dataset, node_idx, device, args):
        '''return negative energy, a vector for all input nodes'''
        x_in, edge_index_in = train_set.x.to(device), train_set.edge_index.to(device)
        x_test, edge_index_test = test_set.x.to(device), test_set.edge_index.to(device)
        print(test_set)
        logits_test = self.encoder(x_test, edge_index_test)
        print("logits test after encoder, ", logits_test)
        train_y = train_set.y[train_idx].to(device)
        logits_test = logits_test[node_idx].to(device)
        num_clusters = train_y.unique(return_counts=True)[0].shape[0]
        cluster_centers = cluster_centers.to(device)
        print("cluster_centers", cluster_centers.shape)
        D = (torch.ones(logits_test.shape[0])*500000).to(device)
        print("logits_test: ", logits_test)
        for i in range(num_clusters):
            whatever = (logits_test.to(device) - cluster_centers[i, :])**2
            if i == 10:
                print("i= 10:", logits_test.to(device) - cluster_centers[i, :])
            temp = torch.sqrt(torch.sum(whatever, dim=1))
            D = torch.vstack((D, temp))
        D = torch.min(torch.t(D), dim=1).values
        print("D is", D.shape, D)
        return D.cpu()
        # return torch.tensor(scores_in)


    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        if len(e.shape) == 1:
            e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        # print('e.shape', e.shape)
        if e.shape[1] == 1:
            return e.squeeze(1)
        else:
            return e
        
    def dist_counting_vector(self,x, cluster_centers, num_clusters, is_ood, device):
        dist_min_ind = torch.ones(x.shape[0])*50000000
        dist_min_ind = dist_min_ind.to(device)
        dist_max_ood = torch.zeros(x.shape[0]).to(device)
        temp = 0
        if is_ood: # making ood_x away from every center
            for i in range(num_clusters):
                dist_max_ood += torch.sqrt(torch.sum((x - cluster_centers[i, :])**2, dim=1)).to(device)       
            dist = dist_max_ood/num_clusters
            # print("current dis_max_ood:", dist[0:10])
            return dist
        else: # making ind_x close to the closest center
            for i in range(num_clusters):
                temp = torch.sqrt(torch.sum((x - cluster_centers[i, :])**2, dim=1)).to(device)
                temp2 = torch.t(torch.vstack((temp, dist_min_ind)))
                dist_min_ind = torch.min(temp2, dim=1).values
            dist = dist_min_ind
            # print("current dist_min_ind:", dist[0:10])

            return dist
    
    
    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)

        # get predicted logits from gnn classifier
        logits_in = self.encoder(x_in, edge_index_in)
        logits_out = self.encoder(x_out, edge_index_out)

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        train_y = dataset_ind.y[train_in_idx]
        num_clusters = train_y.unique(return_counts=True)[0].shape[0]

        if args.use_logits_prop:

            logits_in = self.propagation(logits_in, edge_index_in, args.K, args.alpha)
            logits_out = self.propagation(logits_out, edge_index_out, args.K, args.alpha)

        logits_in_train = logits_in[train_in_idx]
        logits_out_train = logits_out[train_ood_idx]

        cluster_ids_x, cluster_centers = kmeans(X=logits_in_train, num_clusters=num_clusters, distance='euclidean', 
                                                device=torch.device('cuda:'+ str(args.device)))

        ind_dist = self.dist_counting_vector(logits_in_train.to(device), cluster_centers.to(device), num_clusters, False, device)
        ood_dist = self.dist_counting_vector(logits_out_train.to(device), cluster_centers.to(device), num_clusters, True, device)

        # compute supervised training loss
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))
            # print("sup_loss", sup_loss)
        if args.use_reg: # if use energy regularization
            if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
                logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
                logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
                energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
                energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
            else: # for single-label multi-class classification
                energy_in = ind_dist
                energy_out = ood_dist
                # print("energy_in ", energy_in)
                # print("energy_out ", energy_out)
            
            if args.use_dist_prop:
                energy_in = self.propagation(energy_in, edge_index_in, args.K, args.alpha)
                energy_out = self.propagation(energy_out, edge_index_out, args.K, args.alpha)

            # truncate to have the same length
            min_n = min(energy_in.shape[0], energy_out.shape[0])
            if energy_in.shape[0] != energy_out.shape[0]:
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]

            reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)
            # print(reg_loss)
            loss = sup_loss + args.lamda * reg_loss
        else:
            loss = sup_loss
        return loss, cluster_centers


class MSP(nn.Module):
    def __init__(self, d, c, args):
        super(MSP, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError
    
    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):
        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        # print(self.encoder.convs[0].lin.weight.shape)

        if args.dataset in ('proteins', 'ppi'):
            pred = torch.sigmoid(logits).unsqueeze(-1)
            pred = torch.cat([pred, 1- pred], dim=-1)
            max_sp = pred.max(dim=-1)[0]
            return max_sp.sum(dim=1)
        else:
            sp = torch.softmax(logits, dim=-1)
            # print(sp.shape, sp.max(dim=1)[0].shape)
            # print(sp.max(dim=1)[0][:5])
            return sp.max(dim=1)[0]
        
    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_idx = dataset_ind.splits['train']
        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_idx]
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        return loss, torch.zeros((1,1))

class OE(nn.Module):
    def __init__(self, d, c, args):
        super(OE, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            pred = torch.sigmoid(logits).unsqueeze(-1)
            pred = torch.cat([pred, 1- pred], dim=-1)
            max_logits = pred.max(dim=-1)[0]
            return max_logits.sum(dim=1)
        else:
            return logits.max(dim=1)[0]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_in_idx]
        logits_out = self.encoder(dataset_ood.x.to(device), dataset_ood.edge_index.to(device))[train_ood_idx]

        train_idx = dataset_ind.splits['train']
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        loss += 0.5 * -(logits_out.mean(1) - torch.logsumexp(logits_out, dim=1)).mean()
        return loss, torch.zeros((1,1))

class ODIN(nn.Module):
    def __init__(self, d, c, args):
        super(ODIN, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        odin_score = self.ODIN(dataset, node_idx, device, args.T, args.noise)
        return torch.Tensor(-np.max(odin_score, 1))

    def ODIN(self, dataset, node_idx, device, temper, noiseMagnitude1):
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        data = dataset.x.to(device)
        data = Variable(data, requires_grad=True)
        edge_index = dataset.edge_index.to(device)
        outputs = self.encoder(data, edge_index)[node_idx]
        criterion = nn.CrossEntropyLoss()

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        # Using temperature scaling
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
        loss = criterion(outputs, labels)

        datagrad = autograd.grad(loss, data)[0]
        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(datagrad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        '''gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
        gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
        gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)'''
        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))

        # Adding small perturbations to images
        tempInputs = torch.add(data.data, -noiseMagnitude1, gradient)
        outputs = self.encoder(Variable(tempInputs), edge_index)[node_idx]
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        return nnOutputs

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_idx = dataset_ind.splits['train']
        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_idx]
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        return loss, torch.zeros((1, 1))


# noinspection PyUnreachableCode
class Mahalanobis(nn.Module):
    def __init__(self, d, c, args):
        super(Mahalanobis, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, train_set, train_idx, test_set, node_idx, device, args):
        temp_list = self.encoder.feature_list(train_set.x.to(device), train_set.edge_index.to(device))[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        # print('get sample mean and covariance', count)
        num_classes = len(torch.unique(train_set.y))
        sample_mean, precision = self.sample_estimator(num_classes, feature_list, train_set, train_idx, device)
        in_score = self.get_Mahalanobis_score(test_set, node_idx, device, num_classes, sample_mean, precision, count-1, args.noise)
        return torch.Tensor(in_score)

    def get_Mahalanobis_score(self, test_set, node_idx, device,  num_classes, sample_mean, precision, layer_index, magnitude):
        '''
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index
        '''
        self.encoder.eval()
        Mahalanobis = []

        data, target = test_set.x.to(device), test_set.y[node_idx].to(device)
        edge_index = test_set.edge_index.to(device)
        data, target = Variable(data, requires_grad=True), Variable(target)

        out_features = self.encoder.intermediate_forward(data, edge_index, layer_index)[node_idx]
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        datagrad = autograd.grad(loss,data)[0]

        gradient = torch.ge(datagrad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        '''gradient.index_copy_(1, torch.LongTensor([0]).to(device),
                     gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).to(device),
                     gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).to(device),
                     gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7 / 255.0))'''

        tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = self.encoder.intermediate_forward(tempInputs, edge_index, layer_index)[node_idx]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())

        return np.asarray(Mahalanobis, dtype=np.float32)

    def sample_estimator(self, num_classes, feature_list, dataset, node_idx, device):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                 precision: list of precisions
        """
        import sklearn.covariance

        self.encoder.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct = 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        total = len(node_idx)
        output, out_features = self.encoder.feature_list(dataset.x.to(device), dataset.edge_index.to(device))
        output = output[node_idx]

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        target = dataset.y[node_idx].to(device)
        equal_flag = pred.eq(target).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(total):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            precision.append(temp_precision)

        # print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

        return sample_class_mean, precision

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_idx = dataset_ind.splits['train']
        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_idx]
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        return loss, torch.zeros((1,1))