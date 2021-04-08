import torch_geometric

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim, heads=args.heads))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim, heads=args.heads))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb
        print("Dims:", input_dim, hidden_dim, output_dim, args.heads)

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT
        elif model_type == 'torch_geometric_graph_sage':
          def sgconv(in_channels, out_channels, **kwargs):
            return pyg_nn.SAGEConv(in_channels, out_channels, normalize=True)
          return sgconv
        elif model_type == 'torch_geometric_gat':
          def gatconv(in_channels, out_channels, heads=1):
             return pyg_nn.GATConv(in_channels, out_channels, heads)
          return gatconv

    def forward(self, data, edge_index, **kwargs):
        x = data
          
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        x = self.post_mp(x)

        if self.emb == True:
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

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

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


    def aggregate(self, inputs, index, dim_size = None):

        out = torch_scatter.scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
    
        return out


class GraphSage(MessagePassing):
    
    def __init__(self, in_channels, out_channels, heads = 1, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = None
        self.lin_r = None

        ############################################################################
        # TODO: Your code here! 
        # Define the layers needed for the message and update functions below.
        # self.lin_l is the linear transformation that you apply to embedding 
        #            for central node.
        # self.lin_r is the linear transformation that you apply to aggregated 
        #            message from neighbors.
        # Our implementation is ~2 lines, but don't worry if you deviate from this.
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)
        ############################################################################

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        """"""

        out = None

        ############################################################################
        # TODO: Your code here! 
        # Implement message passing, as well as any post-processing (our update rule).
        # 1. First call propagate function to conduct the message passing.
        #    1.1 See there for more information: 
        #        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        #    1.2 We use the same representations for central (x_central) and 
        #        neighbor (x_neighbor) nodes, which means you'll pass x=(x, x) 
        #        to propagate.
        # 2. Update our node embedding with skip connection.
        # 3. If normalize is set, do L-2 normalization (defined in 
        #    torch.nn.functional)
        # Our implementation is ~5 lines, but don't worry if you deviate from this.
        z = self.propagate(edge_index, x=(x, x))
        z1 = self.lin_l(x) + self.lin_r(z)
        if self.normalize:
            z1 = F.normalize(z1, p=2, dim=1)
        out = z1
        ############################################################################

        return out

    def message(self, x_j):

        out = None

        ############################################################################
        # TODO: Your code here! 
        # Implement your message function here.
        # Our implementation is ~1 lines, but don't worry if you deviate from this.
        out = x_j

        ############################################################################

        return out

    def aggregate(self, inputs, index, dim_size = None):

        out = None

        # The axis along which to index number of nodes.
        node_dim = self.node_dim

        ############################################################################
        # TODO: Your code here! 
        # Implement your aggregate function here.
        # See here as how to use torch_scatter.scatter: 
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
        # Our implementation is ~1 lines, but don't worry if you deviate from this.
        out = torch_scatter.scatter(inputs, index=index, dim=node_dim, reduce='mean')
        ############################################################################

        return out


# import torch.optim as optim

# def build_optimizer(args, params):
    # weight_decay = args.weight_decay
    # filter_fn = filter(lambda p : p.requires_grad, params)
    # if args.opt == 'adam':
        # optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    # elif args.opt == 'sgd':
        # optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    # elif args.opt == 'rmsprop':
        # optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    # elif args.opt == 'adagrad':
        # optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    # if args.opt_scheduler == 'none':
        # return None, optimizer
    # elif args.opt_scheduler == 'step':
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    # elif args.opt_scheduler == 'cos':
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    # return scheduler, optimizer


# import time

# import networkx as nx
# import numpy as np
# import torch
# import torch.optim as optim

# from torch_geometric.datasets import TUDataset
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import DataLoader

# import torch_geometric.nn as pyg_nn

# import matplotlib.pyplot as plt


# def train(dataset, args):
    
    # print("Node task. test set size:", np.sum(dataset[0]['train_mask'].numpy()))
    # test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # build model
    # model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            # args)
    # scheduler, opt = build_optimizer(args, model.parameters())

    #train
    # losses = []
    # test_accs = []
    # for epoch in range(args.epochs):
        # total_loss = 0
        # model.train()
        # for batch in loader:
            # opt.zero_grad()
            # pred = model(batch)
            # label = batch.y
            # pred = pred[batch.train_mask]
            # label = label[batch.train_mask]
            # loss = model.loss(pred, label)
            # loss.backward()
            # opt.step()
            # total_loss += loss.item() * batch.num_graphs
        # total_loss /= len(loader.dataset)
        # losses.append(total_loss)

        # if epoch % 10 == 0:
          # test_acc = test(test_loader, model)
          # test_accs.append(test_acc)
        # else:
          # test_accs.append(test_accs[-1])
    # return test_accs, losses

# def test(loader, model, is_validation=True):
    # model.eval()

    # correct = 0
    # for data in loader:
        # with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            # pred = model(data).max(dim=1)[1]
            # label = data.y

        # mask = data.val_mask if is_validation else data.test_mask
        #node classification: only evaluate on nodes in test set
        # pred = pred[mask]
        # label = data.y[mask]
            
        # correct += pred.eq(label).sum().item()

    # total = 0
    # for data in loader.dataset:
        # total += torch.sum(data.val_mask if is_validation else data.test_mask).item()
    # return correct / total
  
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def main():
    for args in [
        {'model_type': 'GraphSage', 'dataset': 'cora', 'num_layers': 2, 'heads': 1, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
    ]:
        args = objectview(args)
        for model in ['GraphSage', 'GAT']:
            args.model_type = model

            # Match the dimension.
            if model == 'GAT':
              args.heads = 2
            else:
              args.heads = 1

            if args.dataset == 'cora':
                dataset = Planetoid(root='/tmp/cora', name='Cora')
            else:
                raise NotImplementedError("Unknown dataset") 
            test_accs, losses = train(dataset, args) 

            print("Maximum accuracy: {0}".format(max(test_accs)))
            print("Minimum loss: {0}".format(min(losses)))

            plt.title(dataset.name)
            plt.plot(losses, label="training loss" + " - " + args.model_type)
            plt.plot(test_accs, label="test accuracy" + " - " + args.model_type)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()

