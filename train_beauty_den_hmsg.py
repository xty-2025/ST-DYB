# -*- encoding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/02/20 10:25:13
@Author  :   Fei gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import argparse
import networkx as nx
import numpy as np
import os
import dgl
import time
import pickle as pkl
import scipy
from torch.utils.data import DataLoader
from utilsxzz10_26 import array_to_matrix, load_adj, get_item_feature, load_feature, get_feature, evaluate, recomm, \
    matrix_to_tensor

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils.minibatch import MyDataset
from utils.utilities import to_device
from eval.link_prediction_init import evaluate_classifier
#from models_GAT_GAT.model import DySAT
from models_GAT_GAT.model import LPreplus
from tqdm import tqdm
import torch
from model_view.MulVDH import MulVDH

torch.autograd.set_detect_anomaly(True)


def inductive_graph(graph_former, graph_later):
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG


def get_test_graph(test_pre_graph_name, test_next_graph_name):
    graphs = []
    with open(test_pre_graph_name, "rb") as f:
        graph = np.load(f)
        graphs.append(graph)
    with open(test_next_graph_name, "rb") as f:
        graph = np.load(f)
        graphs.append(graph)
    return graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=16,
                        help="total time steps used for train, eval and test")
    parser.add_argument('--topk', type=int, nargs='?', default=10,
                        help="topk num")
    parser.add_argument('--total_node', type=int, nargs='?', default=800,
                        help="total node")
    parser.add_argument('--user_num', type=int, nargs='?', default=190,
                        help="user_num")
    parser.add_argument('--HMSG_graph', type=str, nargs='?', default='../dynhen+HMSG/data/beauty/',
                        help='HMSG_graph path')
    parser.add_argument('--graph_name', type=str, nargs='?',
                        default='../dynhen+HMSG/data/beauty/graphs.pkl',
                        help='user graph name')
    parser.add_argument('--item_graph_name', type=str, nargs='?',
                        default='../dynhen+HMSG/data/beauty/item_graphs.pkl',
                        help='item graph name')
    parser.add_argument('--total_graph_name', type=str, nargs='?',
                        default='../dynhen+HMSG/data/beauty/total_graphs.pkl',
                        help='total graph name')
    parser.add_argument('--featuredata_name', type=str, nargs='?', default='features_yelp.npz',
                        help='featuredata name')
    parser.add_argument('--useradj_name', type=str, nargs='?',
                        default='../dynhen+HMSG/data/beauty/user_adj.pkl',
                        help='useradj name')
    parser.add_argument('--itemadj_name', type=str, nargs='?',
                        default='../dynhen+HMSG/data/beauty/item_adj.pkl',
                        help='itemadj name')
    parser.add_argument('--context_pairs_path', type=str, nargs='?',
                        default='../dynhen+HMSG/data/beauty/context_pairs.pkl',
                        help='result of randomwalk')
    parser.add_argument('--edge_sample_path', type=str, nargs='?',
                        default='../dynhen+HMSG/data/beauty/posandneg_edge.npy',
                        help='')


    parser.add_argument('--device', type=str, nargs='?',
                        default='cpu',
                        help='device')

    parser.add_argument('--hidden_units', type=int, nargs='?', default=16,
                        help='HMSG hidden_units')
    parser.add_argument('--datapath', type=str, nargs='?', default='../dynhen+HMSG/data/beauty',
                        help='datapath')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=500,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=64,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                        help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=200,
                        help="patient")
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    parser.add_argument('--gcnresidual', type=bool, nargs='?', default=False,
                        help='Use residual')
    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    # Walk length for random walk sampling.
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.005,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate')
    parser.add_argument('--num_heads', type=int, nargs='?', default=8,
                        help='num_heads')
    parser.add_argument('--GCN_dropout', type=float, default=0.5,
                        help='GCN_Dropout rate (1 - keep probability).')
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.001,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
    parser.add_argument('--GCN_layer_config', type=str, nargs='?', default='512,128',
                        help='Encoder layer config: # units in each GCN layer')
    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each GAT layer')
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')
    args = parser.parse_args()
    print(args)

    # user_graphs
    with open(args.useradj_name, 'rb') as f:
        adjs = pkl.load(f)
    with open((args.graph_name), "rb") as f:
        graphs = pkl.load(f)
    with open((args.total_graph_name), "rb") as f:
        total_graphs = pkl.load(f)
        
   # item_graphs
    with open(args.itemadj_name, 'rb') as f:
        item_adjs = pkl.load(f)
    with open((args.item_graph_name), "rb") as f:
        item_graphs = pkl.load(f)

    # total adjs
    total_adjs_arr = load_adj(args.datapath)
    total_adjs = array_to_matrix(total_adjs_arr)
    for i in range(len(total_adjs)):
        print(total_adjs[i].shape)

    # get all user-item graph of HMSG graph to 'uesr_item'
    user_item = []
    for i in range(args.time_steps):
        user_item.append(np.load(os.path.join(args.HMSG_graph, 'graph{}.npy'.format(i))))

    #user-item to graph
    hg_user_item = []
    for i in range (args.time_steps):
        hg = dgl.heterograph({
            ('user', 'ui', 'item'): (torch.LongTensor(user_item[i][:, 0]), torch.LongTensor(user_item[i][:, 1])),
            ('item', 'iu', 'user'): (torch.LongTensor(user_item[i][:, 1]), torch.LongTensor(user_item[i][:, 0]))
        })  # define edge types (metagraph) num_edge:102219
        hg_user_item.append(hg)

    # all_nodes = []
    # for node_type in hg_user_item[0].ntypes:
    #     nodes = hg_user_item[0].nodes(node_type)
    #     all_nodes.extend(nodes.tolist())
    # print(all_nodes)

    # one_hot feats
    if args.featureless == True:
        feats = [scipy.sparse.identity(total_adjs[args.time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x
                 in total_adjs if
                 x.shape[0] <= total_adjs[args.time_steps - 1].shape[0]]
    assert args.time_steps <= len(adjs), "Time steps is illegal"
        # last two graphs to test

    #tensor of feature(user and item 190 + 610)
     #trans to tensor type
    feat_hmsg_all = []
    for i in range (len(feats)):
        coo = feats[i].tocoo()
        tensor = torch.zeros(coo.shape)
        tensor[coo.row, coo.col] = torch.FloatTensor(coo.data)
        feat_hmsg_all.append(tensor)


    #feat tensor of each timestep graph
    feat_hmsg = []
    for i in range (len(hg_user_item)):
        user_feat = feat_hmsg_all[i][0: (hg_user_item[i].nodes('user')).shape[0]]
        item_feat = feat_hmsg_all[i][0: (hg_user_item[i].nodes('item')).shape[0]]#every graph item node is defined begin 0, so we get more 190 nodes than the indeed item nodes
        feat_hmsg.append({'user_feat': user_feat, 'item_feat': item_feat})

    in_size = {'user': feat_hmsg[0]['user_feat'].shape[1], 'item': feat_hmsg[0]['item_feat'].shape[1]}

    # node2vec, load each timestep positive and negative edge
    with open((args.context_pairs_path), "rb") as f:
        context_pairs_train = pkl.load(f)
    print('get_context_pairs sucessfual')

    # node list with neighbor node to edge pairs (positive and negative edge)
    with open(args.edge_sample_path, "rb") as f:
        edge_sample_4node = np.load(f, allow_pickle=True)

    #to every timestep, get edge sample of each user node (neighbor is item node) cover pos and neg edge
    # (each node in every timestep has 10 pos sample and 10 neg sample)
    edge_sample_pos_4snap = {key: [] for key in range(0, len(edge_sample_4node[i]['node_1']))}
    edge_sample_neg_4snap = {key: [] for key in range(0, len(edge_sample_4node[i]['node_1']))}

    for i in range(len(edge_sample_4node)):
        for j in range(len(edge_sample_4node[i]['node_1'])):
            for k in range(edge_sample_4node[i]['node_1'][j].shape[0]):
                edge_sample_pos_4snap[j].append([int(edge_sample_4node[i]['node_1'][j][k].item()), int(edge_sample_4node[i]['node_2'][j][k].item())])
                # edge_sample_neg_4snap[j].append([int(edge_sample_4node[i]['node_1'][j][k].item()), int(edge_sample_4node[i]['node_2_neg'][j][k][0].item())])
                for l in range(edge_sample_4node[i]['node_2_neg'][j].shape[1]):
                    edge_sample_neg_4snap[j].append([int(edge_sample_4node[i]['node_1'][j][k].item()), int(edge_sample_4node[i]['node_2_neg'][j][k][l].item())])


    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_evaluation_data(total_graphs)
    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
        len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
        len(test_edges_pos), len(test_edges_neg)))
    test_edges_neg = np.array(test_edges_neg)
    val_edges_neg = np.array(val_edges_neg)
    train_edges_neg = np.array(train_edges_neg)
    for ea1 in test_edges_neg:
        if(ea1[0] > ea1[-1]):
            temp = ea1[0]
            ea1[0] = ea1[-1]
            ea1[-1] = temp
    for ea2 in val_edges_neg:
        if(ea2[0] > ea2[-1]):
            temp = ea2[0]
            ea2[0] = ea2[-1]
            ea2[-1] = temp
    for ea3 in train_edges_neg:
        if(ea3[0] > ea3[-1]):
            temp = ea3[0]
            ea3[0] = ea3[-1]
            ea3[-1] = temp
    np.random.shuffle(test_edges_neg)
    np.random.shuffle(val_edges_neg)
    np.random.shuffle(train_edges_neg)

    # build dataloader and model
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


    model = MulVDH(args,
                   meta_paths=[['ui', 'iu'], ['iu', 'ui'], ['ui'], ['iu']],
                   in_size=in_size,
                   aggre_type='mean').to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    # in training
    best_epoch_val = 0
    patient = 0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        opt.zero_grad()
        loss = model.get_loss(args, edge_sample_pos_4snap, edge_sample_neg_4snap, hg_user_item, feat_hmsg)
        loss.backward()
        opt.step()
        epoch_loss = loss.item()
        model.eval()

        emb_list = []
        item_feat_list = []
       # user_emb = model(args, feed_dict["graphs"], feed_dict["total_graphs"],feed_dict["item_graphs"])[:, -2, :].detach().cpu().numpy()
        #emb_list.extend(user_emb)
        #item_feat = model.item_feat[-2].detach().cpu().numpy()
        #item_feat_list.extend(item_feat)
        #emb = emb_list + item_feat_list
        #emb = np.array(emb)
        emb = model(args, hg_user_item, feat_hmsg)[:, -2, :].detach().cpu().numpy()
        
        
        val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                              train_edges_neg,
                                                              val_edges_pos,
                                                              val_edges_neg,
                                                              test_edges_pos,
                                                              test_edges_neg,
                                                              emb,
                                                              emb)
        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]


        if epoch_auc_val > best_epoch_val:
            best_epoch_val = epoch_auc_val
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.3f} Test AUC {:.3f} ".format(epoch,np.mean(epoch_loss),epoch_auc_val,epoch_auc_test))
         # Test Best Model
    model.load_state_dict(torch.load("./model_checkpoints/model_small_dataset.pt"))
    model.eval()

    emb = model(args, hg_user_item, feat_hmsg)[:, -2, :].detach().cpu().numpy()  # 190*16*128


    val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                          train_edges_neg,
                                                          val_edges_pos,
                                                          val_edges_neg,
                                                          test_edges_pos,
                                                          test_edges_neg,
                                                          emb,
                                                          emb)
    auc_val = val_results["HAD"][1]
    auc_test = test_results["HAD"][1]
    

    print("Best Test AUC = {:.3f}".format(auc_test))












