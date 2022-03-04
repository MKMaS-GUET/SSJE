import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation = F.relu,
                 aggr_neighbor_method = "mean",
                 aggr_hidden_method = "sum"):
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_feature, neighbors_node_features):
        neighbor_hidden = self.aggregator(neighbors_node_features)
        self_hidden = torch.matmul(src_node_feature, self.weight)
        neighbor_hidden = torch.unsqueeze(neighbor_hidden,1)
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden
    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method
        )

class GraphSage(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, input_dim = 768 , hidden_dim = [768, 768],
                 num_neighbors_list=[10, 10]):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neigbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
                self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
        # self.layers = num_layers
        # self.emb_dim = emb_dim
        # self.out_dim = emb_dim
        # # gcn layer
        # self.W = nn.ModuleList()
        # for layer in range(self.layers):
        #     input_dim = self.emb_dim if layer == 0 else self.out_dim
        #     self.W.append(nn.Linear(input_dim, input_dim))
        # self.gcn_drop = nn.Dropout(gcn_dropout)


    def forward(self, adj,input):
        # gcn layer
        # denom = adj.sum(2).unsqueeze(2) + 1
        # mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # for l in range(self.layers):
        #     Ax = adj.bmm(inputs)
        #     AxW = self.W[l](Ax)
        #     AxW = AxW + self.W[l](inputs)  # self loop
        #     AxW = AxW / denom
        #     gAxW = F.relu(AxW)
        #     inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        # return inputs,mask

        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        node_features_list = []
        # extend two same data
        node_features_list.append(input)
        node_features_list.append(input)
        node_features_list.append(input)
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                # neighbor_node_features = hidden[hop+1].view(src_node_num, self.num_neigbors_list[hop], -1)
                h = gcn(src_node_features, src_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], mask

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neigbors_list
        )
