# -*- encoding: utf-8 -*-
'''
@File    :   RGCN_model.py
@Time    :   2021/02/19 21:10:00
@Author  :   Fei gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy
# !/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss

from models_GAT_GAT.layers import StructuralAttentionLayer, TemporalAttentionLayer, GCNLayer, InteractionModule
from models_GAT_GAT.TimeNet import TimeNet
from models_GAT_GAT.dida import DGNN
from sklearn.metrics import roc_auc_score
from utils.utilities import fixed_unigram_candidate_sampler


class DyT(nn.Module):
    def __init__(self, args, alpha_init=1.0):
        super(DyT, self).__init__()
        self.args = args

        self.structural_layer_config = list(map(int, self.args.structural_layer_config.split(",")))
        num_features = self.structural_layer_config[-1]
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # return torch.tanh(self.alpha * x) * self.weight.view(1, -1) + self.bias.view(1, -1)

        return torch.tanh(self.alpha * x) * self.weight.view(1, -1)


class LPreplus(nn.Module):
    def __init__(self, args, num_features, time_length):

        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(LPreplus, self).__init__()
        self.args = args
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(time_length, args.window + 1)  # window = 0 => only self.
        self.num_features = num_features
        self.mono_weight_u = nn.Parameter(torch.rand(self.num_time_steps), requires_grad=True)
        self.mono_weight_i = nn.Parameter(torch.rand(self.num_time_steps), requires_grad=True)

        self.self_four_weight = nn.Parameter(torch.rand(self.num_time_steps), requires_grad=True)

        self.GCN_layer_config = list(map(int, args.GCN_layer_config.split(",")))
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.GCN_dropout = args.GCN_dropout
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        self.item_feat = []
        self.GCN, self.structural_attn, self.temporal_attn, self.time_emb = self.build_model()
        self.interaction = InteractionModule(self.temporal_layer_config[-1], self.temporal_layer_config[-1])

        if args.backbone == 'dida':
            self.backbone = DGNN(args)

        # decoder
        self.cs_decoder = self.backbone.cs_decoder
        self.ss_decoder = self.backbone.ss_decoder

        # self.interaction = InteractionModule(self.structural_layer_config[-1], self.structural_layer_config[-1])
        # self.interaction = nn.ModuleList([InteractionModule(self.structural_layer_config[-1], self.structural_layer_config[-1]) for _ in range(self.num_time_steps)])
        self.bceloss = BCEWithLogitsLoss()
        # self.tanh_u = nn.Tanh()
        # self.tanh_i = nn.Tanh()
        # self.tanh = nn.ModuleList([DyT(self.args) for _ in range(self.num_time_steps)])
        # self.tanh_u = DyT(self.args)
        # self.tanh_i = DyT(self.args)
        self.tanh = DyT(self.args)
        self.tanh_fuse = DyT(self.args)


        # self.tanh_i = nn.ModuleList([DyT(self.args) for _ in range(self.num_time_steps)])

    def forward(self, args, user_graphs, total_graphs, item_graphs):

        # HGCN forward

        GCN_out = []
        for t in range(0, self.num_time_steps):
            # first gene mono

            GCN_out_t = self.GCN([total_graphs[t].x, total_graphs[t].adj])

            GCN_out.append(GCN_out_t)
        GCN_outputs = GCN_out
        # split user feats and item feats
        GCN_out_u = []
        GCN_out_i = []
        user_num = args.user_num
        for out in GCN_outputs:
            GCN_out_u.append(out[0:user_num])
            GCN_out_i.append(out[user_num:len(out)])

        # self.item_feat= GCN_out_i      #feats of items(all timestemps)
        GCN_outputs = torch.stack(GCN_outputs)

        # Muti-head Het Attention forward
        structural_user_out = []
        structural_item_out = []

        for t in range(0, self.num_time_steps):
            input_u = total_graphs[t].x[0:user_num]
            input_i = total_graphs[t].x[user_num:]
            user_group = []
            user_group.append(input_u)
            user_group.append(user_graphs[t])
            item_group = []
            item_group.append(input_i)
            item_group.append(item_graphs[t])
            structural_user_out_t = self.structural_attn._modules['structural_layer_0'](user_group)
            structural_item_out_t = self.structural_attn._modules['structural_layer_0'](item_group)
            if len(self.structural_attn) > 1:
                input_u = structural_user_out_t.x
                input_i = structural_item_out_t.x
                user_group = []
                user_group.append(input_u)
                user_group.append(user_graphs[t])
                item_group = []
                item_group.append(input_i)
                item_group.append(item_graphs[t])
                structural_user_out_t = self.structural_attn._modules['structural_layer_1'](user_group)
                structural_item_out_t = self.structural_attn._modules['structural_layer_1'](item_group)
            structural_user_out.append(structural_user_out_t)
            structural_item_out.append(structural_item_out_t)

        mono_user_out = []
        mono_item_out = []
        for t in range(0, self.num_time_steps):
            mono_user_out.append(self.tanh(
                self.mono_weight_u[t] * GCN_out_u[t] + (1 - self.mono_weight_u[t]) * structural_user_out[t].x))
            mono_item_out.append(self.tanh(
                self.mono_weight_i[t] * GCN_out_i[t] + (1 - self.mono_weight_i[t]) * structural_item_out[t].x))
            # mono_user_out.append(self.interaction[t](GCN_out_u[t], structural_user_out[t].x))
            # mono_item_out.append(self.interaction[t](GCN_out_i[t], structural_item_out[t].x))

        # padding outputs along with Ni
        mono_user_outputs = [x[:, None, :] for x in mono_user_out]
        mono_item_outputs = [x[:, None, :] for x in mono_item_out]
        # user
        maximum_node_num = mono_user_outputs[-1].shape[0]
        out_user_dim = mono_user_outputs[-1].shape[-1]
        structural_outputs_user_padded = []
        for out in mono_user_outputs:
            zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_user_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_user_padded.append(padded)
        structural_outputs_user_padded = torch.cat(structural_outputs_user_padded, dim=1)  # 190*16*16

        # item
        maximum_node_num = mono_item_outputs[-1].shape[0]
        out_item_dim = mono_item_outputs[-1].shape[-1]
        structural_outputs_item_padded = []
        for out in mono_item_outputs:
            zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_item_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_item_padded.append(padded)
        structural_outputs_item_padded = torch.cat(structural_outputs_item_padded, dim=1)  # 610*16*16



        # Muti-head Temporal Attention forward

        structural_outputs_user = []
        structural_outputs_item = []
        for t in range(0, self.num_time_steps):
            structural_outputs_user.append(structural_user_out[t].x)
            structural_outputs_item.append(structural_item_out[t].x)
        structural_outputs_user = torch.transpose(torch.stack(structural_outputs_user), 0, 1)
        structural_outputs_item = torch.transpose(torch.stack(structural_outputs_item), 0, 1)

        # user
        temporal_user_out = self.temporal_attn(structural_outputs_user)
        # item
        temporal_item_out = self.temporal_attn(structural_outputs_item)
        out = torch.cat([temporal_user_out, temporal_item_out], dim=0)  # 800*16*32
        temporal_att_out = out

        structural_outputs = torch.cat([structural_outputs_user, structural_outputs_item], dim=1)  # 16*800*16
        structural_outputs = torch.transpose(structural_outputs, 0, 1)  # 16*800*16

        temp_out_cz, temp_out_sz = self.time_emb(structural_outputs)  # temp_out_cz is 16*800*16 temp_out_sz is 16*800*16

        temp_out_cz = torch.transpose(temp_out_cz, 0, 1)  # 800*16*16
        temp_out_sz = torch.transpose(temp_out_sz, 0, 1)
        # temp_out_cz_u = temp_out_cz[:user_num, :, :]
        # temp_out_cz_i = temp_out_cz[user_num:, :, :]
        # temporal_att_out_u = self.temporal_attn(temp_out_cz_u)
        # temporal_att_out_i = self.temporal_attn(temp_out_cz_i)
        #
        # temporal_att_out = torch.cat([temporal_att_out_u, temporal_att_out_i], dim=0)  # 800*16*16


        # final_out_cz = temp_out_cz
        # final_out_sz = temp_out_sz

        #final_out_cz = self.self_four_weight * temporal_att_out + (1 - self.self_four_weight) * torch.transpose(temp_out_cz, 0, 1)
        #final_out_cz =  torch.transpose(temp_out_cz, 0, 1)
        # final_out_cz = 0.8 * structural_outputs + 0.8 * self.interaction(temporal_att_out, torch.transpose(temp_out_cz, 0, 1))
        final_out_cz = self.tanh_fuse(self.self_four_weight * temporal_att_out + (1 - self.self_four_weight) * temp_out_cz)

        # final_out_cz = temporal_att_out
        #final_out_cz = torch.transpose(final_out_cz, 0, 1)
        final_out_sz = temp_out_sz

        return final_out_cz, final_out_sz #800*16*16

    def build_model(self):
        input_dim = self.num_features
        # 0:GCN Layer
        GCN_layers = nn.Sequential()
        # for i in range(len(self.GCN_layer_config)):
        layer = GCNLayer(input_dim=input_dim,
                         hidden_dim=self.GCN_layer_config[0],
                         output_dim=self.GCN_layer_config[-1],
                         GCN_dropout=self.GCN_dropout)
        GCN_layers.add_module(name="GCN_layer_{}".format(0), module=layer)

        # 1: Structural Attention Layers
        input_dim = self.num_features
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

        Time_emb_layers = nn.Sequential()
        layer = TimeNet(args=self.args)
        Time_emb_layers.add_module(name="Time_emb_layer_{}".format(0), module=layer)

        return GCN_layers, structural_attention_layers, temporal_attention_layers, Time_emb_layers

    def get_final_emb(self, args, feed_dict):
        len_train = args.time_steps
        device = args.device

        node_1, node_2, node_2_neg, graphs, total_graphs, item_graphs, _ = feed_dict.values()
        emb_cz, emb_sz = self.forward(args, graphs, total_graphs, item_graphs)  # [N, T, F] 800*16*16
        emb_cz = torch.transpose(emb_cz, 0, 1)
        emb_sz = torch.transpose(emb_sz, 0, 1)
        intervene_times, la = args.intervene_times, args.intervene_lambda
        # generate index label list
        edge_index = []
        edge_label = []

        # user_num to help get item index
        user_num = args.user_num
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            # get the index flag array

            pos_inx = []
            pos_label = []
            for a, b in zip(node_1[t], node_2[t]):
                pos_inx.append(torch.tensor([int(a), int(b)]))
                pos_label.append([1.0])
            neg_inx = []
            neg_label = []
            for i in range(len(node_2_neg[t])):
                for j in range(len(node_2_neg[t][i])):
                    neg_inx.append(torch.tensor([int(node_1[t][i]), int(node_2_neg[t][i][j])]))
                    neg_label.append([0.0])

            pos_inx = torch.stack(pos_inx)
            neg_inx = torch.stack(neg_inx)
            edge_index_t = torch.cat((pos_inx, neg_inx))

            pos_label_array = [torch.tensor(item) for item in pos_label]
            pos_label_tensor = torch.stack(pos_label_array)
            neg_label_array = [torch.tensor(item) for item in neg_label]
            neg_label_tensor = torch.stack(neg_label_array)
            edge_label_t = torch.cat((pos_label_tensor, neg_label_tensor))

            edge_index.append(torch.transpose(edge_index_t, 0, 1))
            edge_label.append(edge_label_t.squeeze())  # edgenum * 1 -> edgenum

        # edge_4_last = edge_label[-1].shape
        edge_label = torch.cat(edge_label)

        def cal_y(embeddings, decoder):

            preds = torch.tensor([]).to(device)
            for t in range(len_train - 1):  # [T-1]
                z = embeddings[t]
                pred = decoder(z, edge_index[t])  # compute simi of each time edges
                preds = torch.cat([preds, pred])
            return preds

        cy = cal_y(emb_cz, self.cs_decoder)
        sy = cal_y(emb_sz, self.ss_decoder)

        return cy, sy, edge_label, emb_cz, emb_sz

    def get_loss(self, args, feed_dict):
        node_1, node_2, node_2_neg, graphs, total_graphs, item_graphs, _ = feed_dict.values()
        # node_1 of fist node is 233, node_2 is 233, node_2_neg is 233*10

        intervene_times, la = args.intervene_times, args.intervene_lambda
        device = args.device

        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        # run gnn
        cy, sy, edge_label, _, _ = self.get_final_emb(args, feed_dict)
        causal_loss = cal_loss(cy, edge_label)

        if self.args.intervene_times <= 0 or self.args.intervene_lambda <= 0:
            loss = causal_loss

        else:
            env_loss = torch.tensor([]).to(device)
            for i in range(intervene_times):
                s1 = np.random.randint(len(sy))
                s = torch.sigmoid(sy[s1]).detach()
                conf = s * cy
                env_loss = torch.cat([env_loss, cal_loss(conf, edge_label).unsqueeze(0)])
            env_mean = env_loss.mean()
            env_var = torch.var(env_loss * intervene_times)
            penalty = env_mean + env_var
            loss = causal_loss + la * penalty

        return loss

    def predict(self, z, pos_edge_index, neg_edge_index, decoder):
        pos_y = z.new_ones(pos_edge_index.size(1)).to(pos_edge_index.device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(pos_edge_index.device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), 0

    def evaluate(self, train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg,
                 emb):

        train_auc, train_ap = self.predict(emb, train_edges_pos, train_edges_neg, self.cs_decoder)
        val_auc, val_ap = self.predict(emb, val_edges_pos, val_edges_neg, self.cs_decoder)
        test_auc, test_ap = self.predict(emb, test_edges_pos, test_edges_neg, self.cs_decoder)

        return train_auc, train_ap, val_auc, val_ap, test_auc, test_ap

