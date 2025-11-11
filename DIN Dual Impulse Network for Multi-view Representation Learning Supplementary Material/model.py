import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#import pyro
from torch.nn.parameter import Parameter
#from geoopt import Lorentz




class Net1(nn.Module):
    def __init__(self, input_dims, hidden_dims, num_views, num_classes, r_view, r_feat, num_layers, dropout, type, device):
        super(Net1, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.r_view = r_view
        self.r_feat = r_feat

        self.num_views = num_views

        self.num_classes = num_classes

        self.fus_type = type

        self.device = device
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        self.project = nn.ModuleList()


        for i in range(num_views):
            self.project.append(nn.Sequential(nn.Linear(input_dims[i], hidden_dims, bias=False),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(dropout),
                                              nn.Linear(hidden_dims, hidden_dims, bias=False)))

        self.view_attention = ViewAttention(num_views, self.r_view)

        self.feat_attention = FeatAttention(hidden_dims, self.r_feat)

        self.MLP = nn.Sequential(
            nn.Linear(hidden_dims + num_views, int(hidden_dims // 2), bias=False),
            nn.ReLU(),
            nn.Linear(int(hidden_dims // 2), hidden_dims + num_views, bias=False)
        )
        self.sigmoid = nn.Sigmoid()


        if self.fus_type == 'self' or self.fus_type == 'cross':
            self.classifier = GCN(hidden_dims + num_views, num_classes, num_layers, dropout)
        elif self.fus_type == 'feat':
            self.classifier = GCN(hidden_dims * 2, num_classes, num_layers, dropout)
        elif self.fus_type == 'view':
            self.classifier = GCN(num_views * 2, num_classes, num_layers, dropout)

    def forward(self, feature_list, adj_hat):

        feats_proj = []
        for i in range(self.num_views):
            feats_proj.append(self.project[i](feature_list[i]))

        h_stack = torch.stack(feats_proj, dim=2)

        N, D, V = h_stack.shape


        pool_view = h_stack.mean(dim=1, keepdim=True)
        pool_feat = h_stack.mean(dim=2, keepdim=True)

        view_x = self.view_attention(pool_view)
        feat_x = self.feat_attention(pool_feat)

        VF_x = self.MLP(torch.cat([view_x, feat_x], 1))

        view_x = VF_x[:, :self.num_views].reshape(N, 1, V)
        feat_x = VF_x[:, self.num_views:].reshape(N, D, 1)

        view_weight = self.sigmoid(view_x)
        feat_weight = self.sigmoid(feat_x)

        out1 = h_stack * view_weight
        out2 = h_stack * feat_weight

        if self.fus_type == 'feat':
            out = torch.cat([out1.mean(dim=2, keepdim=False), out2.mean(dim=2, keepdim=False)], dim=1)
            z = self.classifier(out, adj_hat)
        elif self.fus_type == 'view':
            out = torch.cat([out1.mean(dim=1, keepdim=False), out2.mean(dim=1, keepdim=False)], dim=1)
            z = self.classifier(out, adj_hat)
        elif self.fus_type == 'self':
            out = torch.cat([out1.mean(dim=1, keepdim=False), out2.mean(dim=2, keepdim=False)], dim=1)
            z = self.classifier(out, adj_hat)
        elif self.fus_type == 'cross':
            out = torch.cat([out1.mean(dim=2, keepdim=False), out2.mean(dim=1, keepdim=False)], dim=1)
            z = self.classifier(out, adj_hat)

        return z


class ViewAttention(nn.Module):
    def __init__(self, views, reduction):
        super(ViewAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(views, int(views // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(views // reduction), views, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, D, V = x.shape
        x = self.fc(x.view(N, V))

        return x


class FeatAttention(nn.Module):
    def __init__(self, feats, reduction):
        super(FeatAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feats, int(feats // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(feats // reduction), feats, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #
        N, D, V = x.shape
        x = self.fc(x.view(N, D))

        return x


class GCN(nn.Module):
    def __init__(self, hidden_dims, num_classes, num_layers, dropout):
        super(GCN, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.layers = num_layers
        self.gc = nn.ModuleList()
        for i in range(self.layers-1):
            self.gc.append(GraphConvSparse(self.hidden_dims, self.hidden_dims))
        self.gc.append(GraphConvSparse(self.hidden_dims, self.num_classes))

    def forward(self, input, adj):
        h = input
        for n in range(self.layers-1):
            h = self.gc[n](h, adj)
            h = F.dropout(h, p=self.dropout, training=self.training)
        output = self.gc[-1](h, adj)
        return output


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.tanh):
        super(GraphConvSparse, self).__init__()
        self.activation = activation
        self.layer = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, inputs, flt):
        x = inputs
        x = self.layer(torch.spmm(flt, x))
        if self.activation is None:
            return x
        else:
            return self.activation(x)
