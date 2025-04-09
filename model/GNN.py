import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from model.utils import normalize_A, generate_cheby_adj

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.xavier_uniform_(self.weight, gain=math.sqrt(2.0))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output
class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.gc1 = GraphConvolution(self.nfeat, self.nhid)
        self.gc2 = GraphConvolution(self.nhid, self.nhid)


    def forward(self, x, adj):
        out = F.relu(self.gc1(x, adj))
        out = self.gc2(out, adj)

        return out

class GNN(nn.Module):
    def __init__(self, input, num_out):
        #input: features_in
        #num_out: num_features_out
        super(GNN, self).__init__()
        self.gcn = GCN(input, num_out)
        self.BN1 = nn.BatchNorm1d(input)  #对第二维（第一维为batch_size)进行标准化


    def forward(self, x, adj):
        x = self.BN1.to(x.device)(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(adj)
        # result = self.layer1(x, L)
        result = self.gcn(x,L)
        result = result.reshape(x.shape[0], -1)

        return result
