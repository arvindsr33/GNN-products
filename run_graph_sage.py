import load_data
import train
import torch
import torch_geometric.data as tgd
import project_gat

def main():
    data = load_data.load_small()
    print(data.x.shape,  data.x.dtype)
    print(data.y.shape, data.y.dtype)
    print(data.edge_index.shape)
    data_loader = tgd.DataLoader([data])
    split_idx = {
      'train' : data.train_mask,
      'valid' : data.valid_mask,
      'test' : data.test_mask
    }
    
    num_node_features = 100
    size_train = torch.sum(data.train_mask.to(torch.int))
    size_valid = torch.sum(data.valid_mask.to(torch.int))
    size_test = torch.sum(data.test_mask.to(torch.int))
    print("Splits: ", size_train, size_valid, size_test)
    
    num_labels = int(torch.max(data.y))
    print(num_labels)
    
    args = {'model_type': 'GraphSage', 'num_layers': 2, 'heads': 1, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5, 'epochs': 10,
            'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01,
            'use_edge_index': 1, 'eval_small': 1}
    args_obj = project_gat.objectview(args)
    model = project_gat.GNNStack(num_node_features, args_obj.hidden_dim, num_labels, args=args_obj)
    print('created model')
    train.run(model, data_loader, split_idx, extra_args=args)
    

if __name__ == "__main__":
    main()
    