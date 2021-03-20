import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.conv import MessagePassing


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, arg, emb=False):
        super(GNNStack, self).__init__()
        args = objectview(arg)
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb

    def build_conv_model(self, attention_type):
        if attention_type == 'GraphSage':
            return GraphSage
        elif attention_type == 'GAT':
            return GAT

    def forward(self, x, edge_index):
        # x, edge_index = data, kwargs['edge_index']

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        x = self.post_mp(x)

        if self.emb:
            return x

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()
        self.post_mp[0].reset_parameters()
        self.post_mp[2].reset_parameters()


class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = nn.Linear(in_channels, heads * out_channels)
        self.lin_r = self.lin_l

        self.att_r = nn.Parameter(torch.zeros([heads, out_channels, 1], dtype=torch.float))
        self.att_l = nn.Parameter(torch.zeros([heads, out_channels, 1], dtype=torch.float))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):

        H, C = self.heads, self.out_channels

        z1 = self.lin_l(x)
        z2 = self.lin_r(x)
        h1 = z1.reshape([z1.shape[0], H, C])
        h2 = z2.reshape([z2.shape[0], H, C])
        h1e = h1[edge_index[0]]
        h2e = h2[edge_index[1]]

        alpha_l = torch.matmul(self.att_l.reshape([1, H, 1, C]), h1.reshape([h1.shape[0], H, C, 1]))
        alpha_r = torch.matmul(self.att_r.reshape([1, H, 1, C]), h2.reshape([h2.shape[0], H, C, 1]))
        alpha_l = alpha_l.reshape([h1.shape[0], H])
        alpha_r = alpha_r.reshape([h2.shape[0], H])

        z = self.propagate(edge_index, x=(h1, h2), alpha=(alpha_l, alpha_r))
        out = z.reshape([z.shape[0], z.shape[1] * z.shape[2]])

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):

        ax = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        a = pyg_utils.softmax(
            ax,
            index=index, ptr=ptr, num_nodes=size_i)
        a1 = F.dropout(a, p=self.dropout)
        a1 = a1.reshape([a1.shape[0], a1.shape[1], 1])
        out = torch.mul(a1, x_j)

        return out

    def aggregate(self, inputs, index, dim_size=None):

        out = torch_scatter.scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')

        return out


class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):

        z = self.propagate(edge_index, x=(x, x), dim_size=x.shape)

        out = self.lin_l(x) + self.lin_r(z)
        if self.normalize:
            out = F.normalize(out)
        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):

        # The axis along which to index number of nodes.
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index=index, dim=node_dim, reduce='mean', dim_size=dim_size)

        return out


