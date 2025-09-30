import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import softmax
from torch_scatter import scatter
import numpy as np
import torch.nn as nn
import copy

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feats, adj):
        feats = F.dropout(feats, self.dropout)
        sparse_matrix = adj

        indices = torch.from_numpy(
            np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64))
        values = torch.from_numpy(sparse_matrix.data)
        shape = torch.Size(sparse_matrix.shape)
        adj=torch.sparse.FloatTensor(indices, values, shape)
        support = torch.mm(feats, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output