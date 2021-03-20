import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import DataLoader, ClusterLoader, ClusterData
from ogb.nodeproppred import PygNodePropPredDataset
import os

def get_idx_split():
    dataset_name = "ogbn-products"
    dataset = PygNodePropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()
    return split_idx


def get_product_clusters():
    dataset_name = "ogbn-products"
    dataset = PygNodePropPredDataset(name=dataset_name)

    print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

    data = dataset[0]
    print(data)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    data['train_mask'] = train_mask

    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    data['valid_mask'] = val_mask

    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    data['test_mask'] = test_mask

    cluster_data = ClusterData(data, num_parts=15000, save_dir="dataset")
    return cluster_data, dataset, data, split_idx


def small_clusters():
   dataset_name = "small_cluster"
   data = torch.load("dataset/data_0.pt")
   cluster_data = ClusterData(data, num_parts=10, save_dir="dataset")
   return cluster_data


def get_cluster_batches(cluster_data, batch_size):
    loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True, num_workers=1)
    return loader


def load_small():
    data = torch.load("dataset/data_0.pt")
    print(data)
    return data
    

def load_full():
    data = torch.load("dataset/partition_15000.pt")
    return data


def save_batch(loader, name):
    for batch in loader:
        torch.save(batch, os.path.join("dataset", name))
        break


if __name__ == "__main__":
    data = get_product_clusters()
    # save_batch(data)
