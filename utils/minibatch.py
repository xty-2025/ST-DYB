from typing import DefaultDict
from collections import defaultdict
from torch.functional import Tensor
from torch_geometric.data import Data
import torch_geometric as tg
from utils.utilities import fixed_unigram_candidate_sampler
import torch
import numpy as np
import torch_geometric as tg
import scipy.sparse as sp
import random


import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, args, total_graphs, graphs, features,item_adjs,adjs,total_adjs,context_pairs):
        super(MyDataset, self).__init__()
        self.args = args
        self.graphs = graphs
        self.total_graphs = total_graphs
        self.max_positive = args.neg_sample_size
        self.features = [self._preprocess_features(feat) for feat in features]
        self.adjs = [self._normalize_graph_gcn(a)  for a  in adjs]
        self.item_adjs = [self._normalize_graph_gcn(a)  for a  in item_adjs]
        self.total_adjs=[self._normalize_graph_gcn(a)  for a  in total_adjs]
        self.time_steps = args.time_steps
        self.context_pairs = context_pairs
        self.train_nodes = list(self.graphs[self.time_steps-1].nodes())
        self.min_t = max(self.time_steps - self.args.window - 1, 0) if args.window > 0 else 0
        self.degs = self.construct_degs()
        self.pyg_total_graphs=self._build_pyg_total_graphs()
        self.pyg_graphs = self._build_pyg_graphs()
        self.pyg_item_graphs = self._build_pyg_item_graphs()
        # self.pos_item,self.neg_item=self._get_pos_neg_item(total_adjs)#pos_item neg_item of every node
        self.__createitems__()

    def _get_pos_neg_item(self,adjs):
        user_num=self.args.user_num
        total_pos_item=[]
        total_neg_item=[]
        for adj in adjs:
            pos_item = {}
            neg_item = {}
            for i in range(user_num):
                u_pos=[]
                u_neg=[]
                count=0
                for j in range(user_num,adj.shape[1]):
                    if(adj[i][j]!=0):
                        count+=1
                        u_pos.append(j)#j is item_idx user_num dijige
                for k in range(count):
                    j = np.random.randint(user_num,adj.shape[1])
                    while adj[i][j]!=0:
                        j = np.random.randint(user_num,adj.shape[1])
                    u_neg.append(j)
                pos_item[i]=u_pos
                neg_item[i]=u_neg
            total_pos_item.append(pos_item)#zheg shijianbude yonghujiaohu jiedian
            total_neg_item.append(neg_item)
        return total_pos_item,total_neg_item

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  # D-1/2*A*D-1/2
        return adj_normalized

    def _preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation """
        features = np.array(features.todense())
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        #print('preprocess_features,feats type',type(features))
        #features=sp.csr_matrix(features)
        #print('after preprocess_features,feats type',type(features))
        return features

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        # different from the original implementation
        # degree is counted using multi graph
        degs = []
        for i in range(self.min_t, self.time_steps):
            G = self.total_graphs[i]
            deg = []
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))
            degs.append(deg)
        return degs

    def _build_pyg_graphs(self):
        pyg_graphs = []
        for feat,adj in zip(self.features,self.adjs):
            x=torch.Tensor(feat)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x,edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)
        return pyg_graphs

    def _build_pyg_total_graphs(self):
        pyg_graphs = []
        for feat, adj in zip(self.features, self.total_adjs):
            x = torch.Tensor(feat)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index, adj=adj)
            pyg_graphs.append(data)
        return pyg_graphs

    def _build_pyg_item_graphs(self):
        pyg_graphs = []
        for feat, adj in zip(self.features, self.item_adjs):
            x = torch.Tensor(feat)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)
        return pyg_graphs

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]

    def __createitems__(self):
        self.data_items = {}
        user_num=self.args.user_num
        save_sample = []

        for node in list(self.graphs[self.time_steps - 1].nodes()):  # all user node
            # print(node)
            # print("^^^")
            feed_dict = {}
            node_1_all_time = []
            node_2_all_time = []
            for t in range(self.min_t, self.time_steps):
                node_1 = []
                node_2 = []
                context_item=[]
                for i in self.context_pairs[t][node]:
                    if (not(i in context_item) and i>=user_num ): #if the edge is not exist, and the node is item node, add it
                        context_item.append(i)
                if len(context_item) > self.max_positive:
                    node_1.extend([node] * self.max_positive)
                    node_2.extend(
                        np.random.choice(context_item, self.max_positive, replace=False))
                else:
                    node_1.extend([node] * len(context_item))
                    node_2.extend(context_item)
                assert len(node_1) == len(node_2)
                node_1_all_time.append(node_1)
                node_2_all_time.append(node_2)

            node_1_list = [torch.LongTensor(node) for node in node_1_all_time]
            node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            node_2_negative = []
            for t in range(len(node_2_list)):
                degree = self.degs[t]
                node_positive = node_2_list[t][:, None]
                node_negative = fixed_unigram_candidate_sampler(true_clasees=node_positive,
                                                                num_true=1,
                                                                num_sampled=self.args.neg_sample_size,
                                                                unique=False,
                                                                distortion=0.75,
                                                                unigrams=degree,
                                                                user_num=user_num)


                #print(node_negative)

                node_2_negative.append(node_negative)
            node_2_neg_list = [torch.LongTensor(node) for node in node_2_negative]
            # 将所有数据都放入feed_dict中
            feed_dict['node_1'] = node_1_list
            feed_dict['node_2'] = node_2_list
            feed_dict['node_2_neg'] = node_2_neg_list
            feed_dict["user_graphs"] = self.pyg_graphs
            feed_dict["total_graphs"]=self.pyg_total_graphs
            feed_dict["item_graphs"] = self.pyg_item_graphs
            self.data_items[node] = feed_dict


            #write the edge samples to file
        #     node_sample = {}
        #     node_sample['node_1'] = node_1_list
        #     node_sample['node_2'] = node_2_list
        #     node_sample['node_2_neg'] = node_2_neg_list
        #
        #     save_sample.append(node_sample)
        # path = str(
        #     r'C:\Users\xingzhezhe\Desktop\Mypapers\Dynamic_Hete_Repre\dynhen+HMSG\data\\beauty\posandneg_edge.npy')
        # with open(path, 'ab') as f:
        #     np.save(f, save_sample)
        # f.close()



    @staticmethod
    def collate_fn(samples):

        batch_dict = {}
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list = []
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            for t in range(len(data_list[0])):
                concate.append(torch.cat([data[t] for data in data_list]))
            batch_dict[key] = concate
        batch_dict["user_graphs"] = samples[0]["user_graphs"]  # graph
        batch_dict["total_graphs"] = samples[0]["total_graphs"]  # graph
        batch_dict["item_graphs"] = samples[0]["item_graphs"]
        return batch_dict

'''@staticmethod
    def collate_fn(samples):
        # all info of nodes [143]: {"node_1", "node_2", "node_2_neg"}
        # node_num=args.neg_sample_size
        batch_dict = {}
        for key in ["node_self", "node_pos", "node_neg"]:
            data_list = []
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            for t in range(len(data_list[0])):
                # eve timestemp
                concate.append(torch.cat([data[t] for data in data_list]))  # all nodes
            batch_dict[key] = concate
        batch_dict["graphs"] = samples[0]["graphs"]
        batch_dict["user_graphs"] = samples[0]["user_graphs"]
        return batch_dict
    '''

'''def __createitems1__(self):
        self.data_items = {}
        sample_num = self.args.neg_sample_size

        for node in list(self.graphs[self.time_steps - 1].nodes()):
            feed_dict = {}
            node_self_all=[]

            node_pos_all_time = []
            node_neg_all_time = []
            for t in range(self.time_steps):
                node_self=[]
                node_self.extend([node] * sample_num)
                node_pos=[]
                node_neg=[]
                if len(self.pos_item[t][node])==0:
                    node_self=[]
                    node_self_all.append(node_self)
                    node_pos_all_time.append(node_pos)
                    node_neg_all_time.append(node_neg)
                    continue

                if(len(self.pos_item[t][node])>sample_num):
                    node_pos.extend(np.random.choice(self.pos_item[t][node], sample_num, replace=False))
                else:
                    node_pos = self.pos_item[t][node]
                    if(len(self.pos_item[t][node])<sample_num):
                        #while len(node_pos) < sample_num:
                        l=len(node_pos)
                        for i in range(l, sample_num):
                                node_pos.append(node_pos[(i-l)%l])
                if (len(self.neg_item[t][node]) > sample_num):
                    node_neg.extend(np.random.choice(self.neg_item[t][node], sample_num, replace=False))
                else:
                    node_neg = self.neg_item[t][node]
                    if(len(node_neg)<sample_num):
                        l = len(node_neg)
                        for i in range(l, sample_num):
                            node_neg.append(node_neg[(i - l) % l])

                assert len(node_pos) == len(node_neg)

                node_self_all.append(node_self)
                node_pos_all_time.append(node_pos)
                node_neg_all_time.append(node_neg)

            node_self_list=[torch.LongTensor(node) for node in node_self_all]
            node_pos_list = [torch.LongTensor(node) for node in node_pos_all_time]
            node_neg_list = [torch.LongTensor(node) for node in node_neg_all_time]


            feed_dict['node_self'] =  node_self_list
            feed_dict['node_pos'] = node_pos_list
            feed_dict['node_neg'] = node_neg_list
            feed_dict["graphs"] = self.pyg_total_graphs
            feed_dict["user_graphs"] = self.pyg_graphs

            self.data_items[node] = feed_dict
            '''

    
