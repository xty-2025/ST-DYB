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
from models_GAT_fourier.model import fourier_alone
from tqdm import tqdm

import torch

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
    parser.add_argument('--time_steps', type=int, nargs='?', default=15,
                        help="total time steps used for train, eval and test")
    parser.add_argument('--topk', type=int, nargs='?', default=10,
                        help="topk num")
    parser.add_argument('--total_node', type=int, nargs='?', default=800,
                        help="total node")
    parser.add_argument('--user_num', type=int, nargs='?', default=190,
                        help="user_num")
    parser.add_argument('--graph_name', type=str, nargs='?',
                        default='../dynhen+HMSG+fourier/data/beauty/graphs.pkl',
                        help='user graph name')
    parser.add_argument('--item_graph_name', type=str, nargs='?',
                        default='../dynhen+HMSG+fourier/data/beauty/item_graphs.pkl',
                        help='item graph name')
    parser.add_argument('--total_graph_name', type=str, nargs='?',
                        default='../dynhen+HMSG+fourier/data/beauty/total_graphs.pkl',
                        help='total graph name')
    parser.add_argument('--featuredata_name', type=str, nargs='?', default='features_yelp.npz',
                        help='featuredata name')
    parser.add_argument('--useradj_name', type=str, nargs='?',
                        default='../dynhen+HMSG+fourier/data/beauty/user_adj.pkl',
                        help='useradj name')
    parser.add_argument('--itemadj_name', type=str, nargs='?',
                        default='../dynhen+HMSG+fourier/data/beauty/item_adj.pkl',
                        help='itemadj name')
    parser.add_argument('--context_pairs_path', type=str, nargs='?',
                        default='../dynhen+HMSG+fourier/data/beauty/context_pairs.pkl',
                        help='result of randomwalk')
    parser.add_argument('--datapath', type=str, nargs='?', default='../dynhen+HMSG+fourier/data/beauty',
                        help='datapath')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=500,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=128,
                        help='Batch size (# nodes)')
    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                        help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=500,
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
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--GCN_dropout', type=float, default=0.5,
                        help='GCN_Dropout rate (1 - keep probability).')
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.1,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
    parser.add_argument('--GCN_layer_config', type=str, nargs='?', default='512,16',
                        help='Encoder layer config: # units in each GCN layer')
    parser.add_argument('--structural_head_config', type=str, nargs='?', default='16,8,8',
                        help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # units in each GAT layer')
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',
                        help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=-1,
                        help='Window for temporal attention (default : -1 => full)')
    #fourier need
    parser.add_argument('--nfeat', type=int, default=16, help='dim of input feature')
    parser.add_argument('--nhid', type=int, default=8, help='dim of hidden embedding')  # 8
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--testlength', type=int, default=0, help='length for test')
    parser.add_argument('--vallength', type=int, default=0, help='length for val')
    parser.add_argument('--device', type=str, default='cpu', help='training device')
    parser.add_argument('--metric', type=str, default='acc')
    parser.add_argument('--backbone', type=str, default="GAT")
    parser.add_argument('--static_conv', type=str, default="GCN")
    parser.add_argument('--spec_filt', type=str, default='mask')
    parser.add_argument('--temporature', type=float, default=1, help='')
    parser.add_argument('--post_gnn', type=int, default=0)
    parser.add_argument('--post_temporal', type=int, default=0)
    parser.add_argument('--ctype', type=int, default=0)
    parser.add_argument('--ltype', type=int, default=0, help='')
    parser.add_argument('--use_filt', type=int, default=1, help='')
    parser.add_argument('--heads', type=int, default=8, help='attention heads.')  # 4
    parser.add_argument('--norm', type=int, default=1)  # define the type of Normalization


    parser.add_argument('--dataset', type=str, default='dblp', help='datasets')
    parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
    parser.add_argument('--sampling_times', type=float, default=1, help='negative sampling times')
    parser.add_argument('--clf_layers', type=int, default=2, help='')
    parser.add_argument('--num_classes', type=int, default=4, help='')
    parser.add_argument('--fourier_inf_alpha', type=float, nargs='?', default=0.5)



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
    # one_hor feats
    if args.featureless == True:
        feats = [scipy.sparse.identity(total_adjs[args.time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x
                 in total_adjs if
                 x.shape[0] <= total_adjs[args.time_steps - 1].shape[0]]
        # last two graphs to test
    print('################feats[0]shape',feats[0].shape)
    print('$$$$$$$$$$$$$$$$feats[0]type',type(feats[0]))

    
    assert args.time_steps <= len(adjs), "Time steps is illegal"
    # node2vec
    with open((args.context_pairs_path), "rb") as f:
        context_pairs_train = pkl.load(f)
    print('get_context_pairs sucessfual')
    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_evaluation_data(total_graphs)
    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
        len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
        len(test_edges_pos), len(test_edges_neg)))
    # build dataloader and model
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    dataset = MyDataset(args, total_graphs, graphs, feats, item_adjs, adjs, total_adjs, context_pairs_train)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=MyDataset.collate_fn)

    model = fourier_alone(args, feats[0].shape[1], args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # in training
    best_epoch_val = 0
    patient = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss = []
        for qq, feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)  # node info
            opt.zero_grad()
            loss = model.get_loss(args, feed_dict)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())

        model.eval()
        emb_list = []
        item_feat_list = []
       # user_emb = model(args, feed_dict["graphs"], feed_dict["total_graphs"],feed_dict["item_graphs"])[:, -2, :].detach().cpu().numpy()
        #emb_list.extend(user_emb)
        #item_feat = model.item_feat[-2].detach().cpu().numpy()
        #item_feat_list.extend(item_feat)
        #emb = emb_list + item_feat_list
        #emb = np.array(emb)
        emb = model(args, feed_dict["graphs"], feed_dict["total_graphs"],feed_dict["item_graphs"])[:, -2, :].detach().cpu().numpy()
        
        
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
    model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    model.eval()

    emb = model(args, feed_dict["graphs"],  feed_dict["total_graphs"],feed_dict["item_graphs"])[:, -2, :].detach().cpu().numpy()  # 190*16*128
    #item_feat = model.item_feat[-2].detach().cpu().numpy()

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












