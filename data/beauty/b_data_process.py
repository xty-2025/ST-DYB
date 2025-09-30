import datetime
import networkx as nx
from collections import defaultdict
import sys
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import pickle

#usernum and businessnum: 190 610 max_date,min_date are: 2014-07-22 2006-02-05 START_DATE END_DATE: 2013-03-30 2014-07-23
# parser = argparse.ArgumentParser()
# parser.add_argument('--hidden', type=int, default=16,
#                     help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')

user_counts = defaultdict(lambda: 0)
biz_counts = defaultdict(lambda: 0)


# coding=utf-8

def gcn_layer(A_hat, D_hat, X, W):
    return D_hat ** -1 * A_hat  * X * W


if __name__ == "__main__":

    user_business_interactions = []
    with open("../beauty/beauty.txt") as f:
        for line in f:
            line = line.split("\n")[0].strip()
            u_id, b_id, date = line.split("\t")
            # date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            user_business_interactions.append((u_id, b_id, date))
    user_business_interactions.sort(key=lambda x: x[-1])
    #print(user_business_interactions)
    print(len(user_business_interactions))
    max_date=(user_business_interactions[-1][2])
    min_date=(user_business_interactions[0][2])
    print('max_date,min_date are:',max_date,min_date)

    START_DATE ="2013-04-01"
    END_DATE = "2014-07-23"#取16个yiyue的时间
    print('START_DATE END_DATE:',START_DATE,END_DATE)
    START_DATE = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    END_DATE = datetime.datetime.strptime(END_DATE, '%Y-%m-%d')
    SLICE_len = 1  # 时间步长为三个月

    for obj in user_business_interactions:
        u_id, b_id, date = obj
        if date < START_DATE:
            continue
        if date > END_DATE:
            break
        user_counts[u_id] += 1
        biz_counts[b_id] += 1
    user_business_frequ=[]
    for obj in user_business_interactions:
        u_id, b_id,date = obj
        # date = datetime.datetime.strptime(date, '%Y-%m-%d')
        if date < START_DATE:
            continue
        if date > END_DATE:
            break
        if biz_counts[b_id] < 30:
            continue
        if user_counts[u_id] < 30:
            continue
        user_business_frequ.append((u_id, b_id, date))
    print(len(user_business_frequ))
    # for obj in user_business_frequ:
    #     u_id, b_id, date = obj
    #     print('b_id',biz_counts[b_id])
    #     print('u_id',user_counts[u_id])

    # 用户id 项目id映射到有序数字
    k = -1
    u_num = 0
    b_num = 0
    d = defaultdict(list)
    p = defaultdict(list)
    for obj in user_business_frequ:
        u_id, b_id, date = obj
        if d[u_id] != -1:
            d[u_id] = -1
        if p[b_id] != -1:
            p[b_id] = -1
    for obj in user_business_frequ:
        u_id, b_id, date = obj
        if date < START_DATE:
            continue
        if date > END_DATE:
            break
        if d[u_id] == -1:
            k += 1
            d[u_id] = k
            u_num += 1
    for obj in user_business_frequ:
        u_id, b_id, date = obj
        if date < START_DATE:
            continue
        if date > END_DATE:
            break
        if p[b_id] == -1:
            k += 1
            p[b_id] = k
            b_num += 1
    print('usernum and businessnum:', u_num, b_num)
    num_nodes = u_num + b_num
    # 邻接矩阵
    adj = np.zeros((16, num_nodes , num_nodes ), dtype=np.int)
    slice_id = 0
    pre = 0
    ctr = 0
    edge_count = np.zeros(16, dtype=int)
    for obj in user_business_frequ:
        u_id, b_id, date = obj
        # print (u_id, b_id, date)
        pre = slice_id

        months_diff = ((date - START_DATE)).days // 30#一yue的时长
        if date > END_DATE:
            break
        if months_diff < 0:
            continue
        slice_id = (months_diff // SLICE_len)  # 第几个时间步里的
        edge_count[slice_id] += 1

        if slice_id == 0:
            adj[slice_id, d[u_id], p[b_id]] += 1
            adj[slice_id, p[b_id], d[u_id]] += 1

        if slice_id > 0:
            # slices_links[slice_id] = nx.MultiGraph()
            if slice_id != pre:
                adj[slice_id] = adj[slice_id - 1]
            # print(d[u_id],d[b_id])
            adj[slice_id, d[u_id], p[b_id]] += 1
            adj[slice_id, p[b_id], d[u_id]] += 1

    for i in range(len(adj)):
        print('num of edge in slice {}:'.format(i), adj[i].sum() / 2)
    # 至此 得到每个时间步的邻接矩阵

    # 使用两层GCN得到每个节点的特征，包括用户以及项目

    # 每个slice 初始化feature one_hot
    # slices_features = defaultdict(lambda: {})
    # for slice_id in range(len(adj)):
    #     slices_features[slice_id] = {}
    #     temp = adj[slice_id].shape[0]
    #     temp = np.identity(temp)
    #     for idx in range(adj[slice_id].shape[0]):
    #         slices_features[slice_id][idx] = temp[idx]
    # slice_features_node = []
    # for i in range(len(adj)):
    #     A = adj[i]
    #     A = np.matrix(A)
    #     I = np.eye(adj[i].shape[0])
    #     I = np.matrix(I)
    #     print("nainaidi")
    #     A_hat = A + I
    #     D_hat = np.array(A_hat.sum(1))
    #     # print(D_hat)
    #     D_hat = np.diagflat(D_hat[:, 0])
    #     D_hat = np.matrix(D_hat)
    #     W_1 = np.random.normal(loc=0, scale=1, size=(adj[i].shape[0], 128))
    #     W_1 = np.matrix(W_1)
    #     W_2 = np.random.normal(loc=0, scale=1, size=(W_1.shape[1], 128))#10.10.19:20 xiugai
    #     W_2 = np.matrix(W_2)
    #     W_3 = np.random.normal(loc=0, scale=1, size=(W_2.shape[1], 128))
    #     W_3 = np.matrix(W_3)
    #     H_1 = gcn_layer(A_hat, D_hat, I, W_1)
    #     # print(A.shape)
    #     # print(I.shape)
    #     # print(D_hat)
    #     # print(D_hat ** -1)
    #     # print(W_1.shape)
    #     for l in range(adj[i].shape[0]):
    #         for k in range(128):
    #             H_1[l, k] = max(0, H_1[l, k])
    #     H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
    #     for l in range(H_1.shape[1]):
    #         for k in range(128):
    #             H_2[l, k] = max(0, H_2[l, k])
    #
    #     slice_features_node.append(H_1)

        # print(slice_features_node)
    for i in range(len(adj)):
        name = 'graph{}.npy'.format(str(i))
        arr = []
        for j in range(u_num):
            for k in range(adj[i].shape[1]):
                if(adj[i][j][k]>0):
                    a = adj[i][j][k]
                    while a>0:
                        arr.append([j, k])
                        a = a-1

        np.save(name, arr)
        with open(name, "rb") as f:
            graph = np.load(f)
            print(graph)
        print("aaaaaaaaaaaaaaaaa")


    # 把邻接矩阵和图特征保存起来
    '''
    s='graphs_yelp.npy'
    for i in range(len(adj)):
        s='a'+s
        np.save(s, adj[i])
        '''




