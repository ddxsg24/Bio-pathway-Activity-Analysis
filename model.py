import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import numpy as np
import random
# 设置随机种子
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class DMGI(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, num_relations):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels, out_channels) for _ in range(num_relations)])
        # self.convs = torch.nn.ModuleList(
        #     [GATConv(in_channels, out_channels, heads=1) for _ in range(num_relations)])
        self.M = torch.nn.Bilinear(out_channels, out_channels, 1)
        self.Z = torch.nn.Parameter(torch.empty(num_nodes, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(self, x, edge_indices):
        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True))


        return pos_hs, neg_hs, summaries

    def loss(self, pos_hs, neg_hs, summaries):
        loss = 0.
        loss1 = 0.
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
            loss1 += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean() #loss1
            loss1 += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean() #loss1
            # loss += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean()
            # loss += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean()

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        #loss += 0.001 * (pos_reg_loss - neg_reg_loss)
        loss = loss1 + 0.001 * (pos_reg_loss - neg_reg_loss) #loss1

        return loss, loss1
