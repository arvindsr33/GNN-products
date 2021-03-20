import torch
from torch_geometric.nn import GCNConv
import graphSAGE
import resnet_postmp as res
import ResidualMP as res_mp


def get_model(args):
    input_dim = args['input_dim']
    output_dim = args['output_dim']
    model_type = args['model_type']

    print(model_type)
    if model_type == 'GraphSage' or model_type == 'GAT':
        return graphSAGE.GNNStack(input_dim, args['hidden_dim'], output_dim, args)
    elif model_type == 'ResPostMP':
        return res.ResNetPostMP(input_dim, args['hidden_dim'], output_dim, args)
    elif model_type == 'ResidualMP':
        return res_mp.ResidualMP(input_dim, args['hidden_dim'], output_dim, args)
    else:
        return GCN(input_dim, args['hidden_dim'], output_dim, args['num_layers'], args['dropout'], args['return_embeds'])


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(GCN, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList([GCNConv(input_dim, hidden_dim)] + \
                                         [GCNConv(hidden_dim, hidden_dim) for i in range(num_layers - 2)] + \
                                         [GCNConv(hidden_dim, output_dim)])

        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim)
                                        for _ in range(num_layers - 1)])

        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        z = self.convs[0](x, adj_t)

        for i, layer in enumerate(self.bns):
            bn = layer(z)
            h = torch.nn.functional.relu(bn)
            d = torch.nn.functional.dropout(h, p=self.dropout)
            z = self.convs[i + 1](d, adj_t)

        if self.return_embeds:
            out = z
        else:
            out = self.softmax(z)

        return out

