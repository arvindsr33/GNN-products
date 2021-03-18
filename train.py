import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import load_data as ld
import models
from tqdm import tqdm
import torch_geometric.transforms as T
import copy


def train(model, data_loader, optimizer, device):
    model.train()

    total_loss = 0
    total_nodes = 0

    with tqdm(total=len(data_loader)) as progress_bar:
        for batch in data_loader:
            batch = batch.to(device)
            if batch.train_mask.sum() == 0:
                continue

            optimizer.zero_grad()
            out = model(batch.x, edge_index=batch.edge_index, adj_t=batch.edge_index)[batch.train_mask]
            y = torch.flatten(batch.y[batch.train_mask])
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_nodes += nodes
            progress_bar.update(1)

    # model.to("cpu")
    return total_loss / total_nodes


@torch.no_grad()
def test(model, data, split_idx, evaluator, use_edge_index=False):
    model.eval()
    print("starting")
    print(data)

    if use_edge_index:
        edge_index = data.edge_index
        print(edge_index.shape)
        
        out = model(data.x, edge_index=edge_index)
    else:
        out = model(data.x, data.adj_t)
    print("Out done")

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    # test_acc = evaluator.eval({
    #     'y_true': data.y[split_idx['test']],
    #     'y_pred': y_pred[split_idx['test']],
    # })['acc']

    return train_acc, valid_acc

def run(model,data_loader,split_idx, extra_args=None):
    device = "cpu"
    args = {
        'device': device,
        'num_layers': 3,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 100,
    }
    if extra_args:
      for k in extra_args:
        args[k] = extra_args[k]

    dataset_name = "ogbn-products"
    if extra_args.get('use_edge_index', 0) == 1:
      dataset_eval = PygNodePropPredDataset(name=dataset_name)
    else:
      dataset_eval = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())

    eval_data = dataset_eval[0]

    eval_split_idx = dataset_eval.get_idx_split()

    evaluator = Evaluator(name='ogbn-products')

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss

    best_model = None
    best_valid_acc = 0

    print("----------------------------------")
    print("Params:", args)
    print("======")
    
    for epoch in range(1, 1 + args["epochs"]):
        model.to(device)
        loss = train(model, data_loader, optimizer, device)
        model.to("cpu")
        if extra_args.get('eval_small', 0) == 1:
          result = test(model, data_loader.dataset[0], split_idx, evaluator, use_edge_index=True)
        else:
          result = test(model, eval_data, eval_split_idx, evaluator, use_edge_index=extra_args['use_edge_index'])
        train_acc, valid_acc = result
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * 0:.2f}%')
    
    
    
if __name__ == "__main__":
    device = "cuda"
    args = {
        'device': device,
        'num_layers': 3,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 100,
    }

    cluster_data, dataset, data, split_idx = ld.get_product_clusters()
    data_loader = ld.get_cluster_batches(cluster_data, 4)

    dataset_name = "ogbn-products"
    dataset_eval = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())
    eval_data = dataset_eval[0]

    eval_split_idx = dataset_eval.get_idx_split()

    model = models.GCN(data.num_features, args['hidden_dim'],
                dataset.num_classes, args['num_layers'],
                args['dropout'])

    evaluator = Evaluator(name='ogbn-products')

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        model.to(device)
        loss = train(model, data_loader, optimizer, device)
        model.to("cpu")
        result = test(model, eval_data, eval_split_idx, evaluator)
        train_acc, valid_acc = result
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * 0:.2f}%')


