import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.conv import MessagePassing
from objectview import objectview


class ResNetBlock(torch.nn.Module):
    def __init__(self, module):
        super(ResNetBlock, self).__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

    def reset_parameters(self):
        for l in self.module:
            l.reset_parameters()


class ResidualMP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, arg, emb=False):
        super(ResidualMP, self).__init__()
        args = objectview(arg)
        self.convs = nn.ModuleList()
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.emb = emb

        # May want to change input/hidden/output dim?
        self.convs.append(DeeperGraphSage(input_dim, hidden_dim, hidden_dim=args.message_hidden, dropout=args.dropout))
        for l in range(args.num_layers-1):
            self.convs.append(
                DeeperGraphSage(hidden_dim, hidden_dim, hidden_dim=args.message_hidden, dropout=args.dropout))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index):

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


class DeeperGraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim, normalize=True, dropout=0.5, bias=False, **kwargs):
        super(DeeperGraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.dropout = dropout

        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)
        # This has the residual of the graphsage message at the output
        # Could look at a deeper version?
        self.mlp = nn.Sequential(
            ResNetBlock(
                nn.Sequential(
                    nn.Linear(out_channels, hidden_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, out_channels)
                )
            ),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):

        z = self.propagate(edge_index, x=(x, x), dim_size=x.shape)

        prop = self.lin_l(x) + self.lin_r(z)
        out = self.mlp(prop)

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