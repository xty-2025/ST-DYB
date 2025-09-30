# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy
#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss

from models_GAT_fourier.layers import StructuralAttentionLayer, TemporalAttentionLayer, GCNLayer
from models_GAT_fourier.TimeNet import TimeNet
from utils.utilities import fixed_unigram_candidate_sampler

class fourier_alone(nn.Module):
    def __init__(self, args, num_features, time_length):

        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(fourier_alone, self).__init__()
        self.args = args
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features
        self.GCN_layer_config= list(map(int, args.GCN_layer_config.split(",")))
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.GCN_dropout = args.GCN_dropout
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        self.item_feat=[]
        self.GCN, self.structural_attn, self.temporal_attn, self.time_emb = self.build_model()
        self.bceloss = BCEWithLogitsLoss()
        self.lin = nn.Linear(args.nfeat, 16, bias=False)

    def forward(self, args, user_graphs,total_graphs,item_graphs):
        # GCN Attention forward

        GCN_out=[]
        for t in range(0,self.num_time_steps):
            GCN_out.append(self.GCN([total_graphs[t].x, total_graphs[t].adj]))
        GCN_outputs = GCN_out
        # split user feats and item feats
        GCN_out_u=[]
        GCN_out_i=[]
        user_num=args.user_num
        for out in GCN_outputs:
            GCN_out_u.append(out[0:user_num])
            GCN_out_i.append(out[user_num:len(out)])

        # self.item_feat= GCN_out_i      #feats of items(all timestemps)

        structural_user_out = []
        structural_item_out = []
        for t in range(0, self.num_time_steps):
            user_group=[]
            user_group.append(GCN_out_u[t])
            user_group.append(user_graphs[t])
            item_group=[]
            item_group.append(GCN_out_i[t])
            item_group.append(item_graphs[t])
            structural_user_out.append(self.structural_attn(user_group))
            structural_item_out.append(self.structural_attn(item_group))
        structural_user_outputs = [g.x[:,None,:] for g in structural_user_out]
        structural_item_outputs = [g.x[:,None,:] for g in structural_item_out]

        # padding outputs along with Ni
        #user
        maximum_node_num = structural_user_outputs[-1].shape[0]
        out_user_dim = structural_user_outputs[-1].shape[-1]
        structural_outputs_user_padded = []
        for out in structural_user_outputs:
            zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_user_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_user_padded.append(padded)
        structural_outputs_user_padded = torch.cat(structural_outputs_user_padded, dim=1)
 
        #item
        maximum_node_num = structural_item_outputs[-1].shape[0]
        out_item_dim = structural_item_outputs[-1].shape[-1]
        structural_outputs_item_padded = []
        for out in structural_item_outputs:
            zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_item_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_item_padded.append(padded)
        structural_outputs_item_padded = torch.cat(structural_outputs_item_padded, dim=1)
        '''
        # Temporal Attention forward
        #user
        temporal_user_out = self.temporal_attn(structural_outputs_user_padded)#190 16 128
        #item
        temporal_item_out = self.temporal_attn(structural_outputs_item_padded)#610 16 128
        #out = torch.cat([temporal_user_out,temporal_item_out],dim=0)
        temporal_user_out = torch.transpose(temporal_user_out, 0, 1)
        temporal_item_out = torch.transpose(temporal_item_out, 0, 1)'''
        # time embedding
        # cz, sz = self.time_emb(temporal_out)  #
        temporal_user_out = torch.transpose(structural_outputs_user_padded, 0, 1)#timestep*user_num*dim
        temporal_item_out = torch.transpose(structural_outputs_item_padded, 0, 1)#timestep*item_num*dim
        temporal_user_resnet = temporal_user_out# timestep*user_num*dim 16 190 32
        temporal_item_resnet = temporal_item_out# timestep*item_num*dim 16 610 32

        temp_user = self.time_emb(temporal_user_out)
        temp_item = self.time_emb(temporal_item_out)
        H_user = temporal_user_resnet + args.fourier_inf_alpha *temp_user #16 190 16 timestep*node_num_dim
        H_item = temporal_item_resnet + args.fourier_inf_alpha *temp_item   # 16 610 16 timestep*node_num*dim

        H = torch.cat([H_user,H_item],dim=1)
        H = torch.transpose(H, 0, 1) #

        return H


    def build_model(self):
        input_dim = self.num_features
        #0:GCN Layer
        GCN_layers=nn.Sequential()
        #for i in range(len(self.GCN_layer_config)):
        layer = GCNLayer(input_dim=input_dim,
                         hidden_dim=self.GCN_layer_config[0],
                         output_dim=self.GCN_layer_config[-1],
                         GCN_dropout=self.GCN_dropout)
        GCN_layers.add_module(name="GCN_layer_{}".format(0), module=layer)
    

        # 1: Structural Attention Layers
        input_dim=self.GCN_layer_config[-1]
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,  # featurs;
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.args.residual)
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]

        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        # Time embedding using fourier
        Time_emb_layers = nn.Sequential()
        layer = TimeNet(args=self.args)
        Time_emb_layers.add_module(name="Time_emb_layer_{}".format(0), module=layer)

        return GCN_layers, structural_attention_layers, temporal_attention_layers, Time_emb_layers

    def get_loss(self, args, feed_dict):
        # print("feed_dict.value:", feed_dict.values())
        # print("dict_len:", len(feed_dict.values()))
        # print("feed_dict.key:", feed_dict.keys())
        node_1, node_2, node_2_neg, graphs, total_graphs, item_graphs, _ = feed_dict.values()
        # run gnn
        emb = self.forward(args, graphs, total_graphs, item_graphs) # [N, T, F]
        #item_emb=self.item_feat
        final_user_emb, final_item_emb = emb.split([args.user_num, args.total_node-args.user_num], dim=0)
   
        #user_num to help get item index
        user_num = args.user_num
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            # get the index flag array
            index_dis_pos = []
            index_dis_pos.extend([user_num] * len(node_2[t])) #[190, 190, ..., 190]
            pos_item_inx = []
            for a, b in zip(node_2[t], index_dis_pos):
                pos_item_inx.append(int(a - b)) #item node index - user_num
            index_dis_neg = torch.zeros_like(node_2_neg[t])
            for i in range(len(node_2_neg[t])):
                for j in range(len(node_2_neg[t][i])):
                    index_dis_neg[i][j] = user_num #[[190, 190, ..., 190], [190, 190, ... 190], ...]
            neg_item_inx = torch.zeros_like(node_2_neg[t])
            for i in range(len(node_2_neg[t])):
                for j in range(len(node_2_neg[t][i])):
                    neg_item_inx[i][j] = node_2_neg[t][i][j] - index_dis_neg[i][j]

            emb_user_t = final_user_emb[:, t, :].squeeze()
            emb_item_t = final_item_emb[:, t, :].squeeze()
            source_node_emb = emb_user_t[node_1[t]]
            tart_node_pos_emb = emb_item_t[pos_item_inx]
            tart_node_neg_emb = emb_item_t[neg_item_inx]
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :] * tart_node_neg_emb, dim=2).flatten()
            # neg_score = -torch.sum(source_node_emb*tart_node_neg_emb, dim=1)
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + self.args.neg_weight * neg_loss
            self.graph_loss += graphloss
        return self.graph_loss



