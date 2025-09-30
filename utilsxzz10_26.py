import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torch_geometric as tg
from collections import defaultdict
from torch_geometric.data import Data


def array_to_matrix(adjs):
    f_adj = []
    for adj in adjs:
        a = sp.csr_matrix(adj)
        f_adj.append(a)
    return f_adj
#adjs to user_graphs

def get_feature(features,start,end):
    #slice_features_node 16073*128
    # slice_user_feature=[]
    # n=user_num
    # d=feature_dim
    # np.zeros((n, d), dtype=np.float32)
    total_feats=[]
    print(len(features))
    for i in range(len(features)):
        total_feature=[]
        for j in range(start,end):
            total_feature.append(features[i][j])
        total_feature=(np.matrix(total_feature))
        total_feature=sp.csr_matrix(total_feature)
        total_feats.append(total_feature)

    return (total_feats)

def get_item_feature(features,start,end):
    item_feats=[]
    for i in range(len(features)):
        item_feature = []
        for j in range(start,end):
            item_feature.append(features[i][j])
        item_feature = (np.matrix(item_feature))
        item_feature = sp.csr_matrix(item_feature)
        item_feats.append(item_feature)
    return item_feats

def load_adj(dataset_str):

    adjs=[]
    s = 'graphs_yelp.npy'
    for i in range(1,17):
        s='a'+s
        with open("{}/{}".format(dataset_str,s), "rb") as f:
            adj = np.load(f)
        adjs.append(adj)

    return adjs

def load_feature(dataset_str,file_name):
    with open ("{}/{}".format(dataset_str, file_name), "rb") as ft:
        features=np.load(ft)
        feature=features['feats']
    return feature




def evaluate(item_rec,test_graphs,user_num,item_num):
    k=10
    pre_graph = test_graphs[-2]
    next_graph = test_graphs[-1]
    real_list = defaultdict(list)
    #get every user's item in last timestep real_list{8: [1, 2, 3, 4]}
    for u in range(0,user_num):
        it=[]
        for i in range(user_num,user_num+item_num):#everytime item
            if((pre_graph[u][i]==0) and (next_graph[u][i]!=0)):
                it.append(i)
        real_list[u]=it
    user_ranks=item_rec
    Hit=0
    NDCG=0
    count = 0
    rank=0
    half_of_user_num = int(user_num / 2)
    for u in range(0, half_of_user_num):
        if (len(real_list[u]) != 0):
            count += 1
            flag=0
            for i in range(len(user_ranks[u])):
                if(user_ranks[u][i]==real_list[u][0]):
                    flag=1
                    rank=i+1
            if flag==1:
                Hit+=1
                NDCG+=1 / np.log2(rank + 2)
    val_n_score = NDCG/ count
    val_h_score = Hit / count

    count = 0
    rank = 0
    for u in range(0, user_num):
        if (len(real_list[u]) != 0):
            count += 1
            flag = 0
            for i in range(len(user_ranks[u])):
                if (user_ranks[u][i] == real_list[u][0]):
                    flag = 1
                    rank = i + 1
            if flag == 1:
                Hit += 1
                NDCG += 1 / np.log2(rank + 2)
    test_n_score = NDCG / count
    test_h_score = Hit / count
    return val_n_score, val_h_score, test_n_score, test_h_score

def recomm(user_feats,item_feat,test_graphs,user_num,item_num,k):
    pre_graph = test_graphs[-2]
    next_graph = test_graphs[-1]
    user_ranks = {}
    rank_score = {}
    for u in range(0, user_num):
        t = {}
        for i in range(user_num, user_num + item_num):
            if (pre_graph[u][i] == 0):
                u_feat = np.matrix(user_feats[u])
                i_feat = np.matrix(item_feat[i - user_num])
                t[i] = u_feat * i_feat.T
        t = sorted(t.items(), key=lambda d: d[1], reverse=True)
        t = dict(t)
        rank_score[u] = t  # {0: [(5, 6), (1, 3), (2, 0)]}
    # print('每个用户对应项目得分是',rank_score)
    for u in range(0, user_num):
        it = []
        ite = []
        for j in rank_score[u].keys():
            it.append(j)
        for l in range(k):
            ite.append(it[l])
        user_ranks[u] = ite

    total_item_rec = {}
    for u in range (user_num):
        u_total_item=[]
        for i in range(0,k):
            u_total_item.append(user_ranks[u][i])
        total_item_rec[u]=u_total_item
    return total_item_rec
def matrix_to_tensor(feats):
    item_feats=[]
    ite_f=[]
    feats_nor=[preprocess_features(feat) for feat in feats]
    for i in range(len(feats_nor)):
        for feat in (feats_nor[i]):
            ite_f.append(feat)
        ite_f_tensor=torch.FloatTensor(ite_f)
        item_feats.append(ite_f_tensor)
    return item_feats

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation """
    features = np.array(features.todense())
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
