import torch
import numpy as np
import torch.nn.functional as F


def masked_mean(_input, _mask):
    # _input: batch x time x hid
    # _mask: batch x time
    _input = _input * _mask.unsqueeze(-1)
    # masked mean
    avg_input = torch.sum(_input, 1)  # batch x enc
    _m = torch.sum(_mask, -1)  # batch
    tmp = torch.eq(_m, 0).float()  # batch
    if avg_input.is_cuda:
        tmp = tmp.cuda()
    _m = _m + tmp
    avg_input = avg_input / _m.unsqueeze(-1)  # batch x enc
    return avg_input


def masked_max(_input, _mask):
    """
    Masked max along dim 1
    :param _input: bz x max_n_node x hidden_size
    :param _mask: bz x max_n_node
    """
    masked_input = torch.where(_mask.byte().unsqueeze(2).repeat(1, 1, _input.size()[2]), _input, -torch.Tensor([float('inf')]).to(_input.device))
    masked_input = torch.max(masked_input, dim=1)[0]
    return masked_input


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, input, adj):
        # input: batch x num_entity x in_dim
        # adj:   batch x num_entity x num_entity
        support = self.weight(input)  # batch x num_entity x out_dim
        output = torch.bmm(adj, support)  # batch x num_entity x out_dim
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class StackedGraphConvolution(torch.nn.Module):
    '''
    input:  entity features:    batch x num_entity x input_dim
            adjacency matrix:   batch x num_entity x num_entity
    output: res:                batch x num_entity x hidden_dim[-1]
    '''

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.0, use_highway_connections=False):
        super(StackedGraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims  # this is a list which contains the hidden dimension of each gcn layer
        self.dropout_rate = dropout_rate
        self.nlayers = len(self.hidden_dims)
        self.stack_rnns()
        self.use_highway_connections = use_highway_connections
        if self.use_highway_connections:
            self.stack_highway_connections()

    def stack_highway_connections(self):
        highways = [torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i]) for i in range(self.nlayers)]
        self.highways = torch.nn.ModuleList(highways)
        self.input_linear = torch.nn.Linear(self.input_dim, self.hidden_dims[0])

    def stack_rnns(self):
        gcns = [GraphConvolution(self.input_dim if i == 0 else self.hidden_dims[i - 1], self.hidden_dims[i])
                for i in range(self.nlayers)]
        self.gcns = torch.nn.ModuleList(gcns)

    def forward(self, x, adj):
        res = x
        for i in range(self.nlayers):
            # highway connection
            if self.use_highway_connections:
                if i == 0:
                    prev = self.input_linear(res)
                else:
                    prev = res.clone()
            res = self.gcns[i](res, adj)  # batch x num_nodes x hid
            res = F.relu(res)
            res = F.dropout(res, self.dropout_rate, training=self.training)
            # highway connection
            if self.use_highway_connections:
                gate = torch.sigmoid(self.highways[i](res))
                res = gate * res + (1 - gate) * prev
        return res
