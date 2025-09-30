# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Time    :   2021/02/18 14:30:13
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import softmax
from torch_scatter import scatter
from models_GAT_GAT_fourier.GCN_layer import GraphConvolution
import numpy as np
import copy


class GCNLayer(nn.Module):
    def __init__(self,
                input_dim,
                hidden_dim,
                output_dim,
                GCN_dropout):
        super(GCNLayer, self).__init__()
        self.dropout = GCN_dropout
        # self.lin = nn.Linear(input_dim,input_dim)
        self.gc1 = GraphConvolution(input_dim, hidden_dim,self.dropout)
        self.gc2 = GraphConvolution(hidden_dim, output_dim,self.dropout)
        self.dropout = GCN_dropout


    def forward(self, group):
        # graph = copy.deepcopy(total_graph)
        feats = group[0]
        adj = group[1]
        # feats = graph.x
        x = F.relu(self.gc1(feats, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class StructuralAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                output_dim, 
                n_heads, 
                attn_drop, 
                ffd_drop,
                residual):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))  # [1, 16, 8]; a1, attention
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))  # [1, 16, 8]; a2, attention

        self.reset_param(self.att_l)
        self.reset_param(self.att_r)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)  # [143, 128]

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self, group):
        graph=group[1]
        feat=group[0]
      
        graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight.reshape(-1, 1)#Auv

        H, C = self.n_heads, self.out_dim
        
        #feats is  user feats from GCN
        x = self.lin(feat).view(-1, H, C) # [N, heads, out_dim]; [18, 143]*[143, 128] => [18,128] => [18,16,8]
        alpha_l = (x * self.att_l).sum(dim=-1).squeeze()
        alpha_r = (x * self.att_r).sum(dim=-1).squeeze()  # a2*X

        alpha_l = alpha_l[edge_index[0]]
        alpha_r = alpha_r[edge_index[1]]
        
        alpha = alpha_r + alpha_l
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)

        coefficients = softmax(alpha, edge_index[1])

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)
        x_j = x[edge_index[0]]

        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))  # [nodes, heads, dim]
        out = out.reshape(-1, self.n_heads*self.out_dim) #[num_nodes, output_dim]
        if self.residual:
            out = out + self.lin_residual(feat)
        graph.x = out
        return graph

        
class TemporalAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # [128, 128]; W*Q
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):

        position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs]

        # 2: Query, Key based multi-head self attention. [143, 16, 128]
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)  # Q*K
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2**32+1)
        outputs = torch.where(masks==0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)  # output:[2288, 16, 16]
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)  # dropout
        outputs = torch.matmul(outputs, v_)
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)


