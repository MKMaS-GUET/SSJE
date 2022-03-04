import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features = 768, out_features = 768, dropout = 0.1, alpha = 0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        # denom = adj.sum(2).unsqueeze(2) + 1
        h = torch.squeeze(h,0)
        adj = torch.squeeze(adj,0)
        # if len(adj.size(0) == 1):
        #     adj = adj.view(adj.size(0) * adj.size(1), adj.size(2)) #adj = torch.squeeze(adj,0)
        # if len(self.W.size()) == 1:
        #     Wh = torch.mm(h, self.W.unsqueeze(dim=0))
        # else:
        Wh = torch.matmul(h, self.W)
        # if len(h.size()) > 2:
        #     h = h.view(h.size(0) * h.size(1), h.size(2))
        # else: h = torch.squeeze(h,0)
        # adj = adj.view(adj.size(0) * adj.size(1), adj.size(2))

        # Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)

        # neighbor_hidden = torch.unsqueeze(neighbor_hidden,1)
        # self.W = torch.unsqueeze(self.W,0)
        # Wh = torch.bmm(h,torch.unsqueeze(self.W,0))
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)

        temp_atten = torch.matmul(adj,e)
        attention = torch.where(temp_atten > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, nfeat = 768, nhid = 8, nclass = 1, dropout = 0.1, alpha= 0.2, nheads = 8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x,adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = torch.unsqueeze(x,0)
        return F.log_softmax(x, dim=1)
