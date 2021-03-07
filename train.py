import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import load_data as ld
import models
from tqdm import tqdm


def train(model, data_loader, optimizer, device):
    model.train()

    total_loss = 0
    total_nodes = 0

    with tqdm(total=len(data_loader)) as progress_bar:
        for batch in data_loader:
            data = batch.to(device)
            if data.train_mask.sum() ==0:
                continue

            optimizer.zero_grad()
            out = model(data.x, data.edge_index)[data.train_mask]
            y = torch.flatten(data.y[data.train_mask])
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_nodes += nodes
            progress_bar.update(1)

    return total_loss / total_nodes


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

    dataset_name = "ogbn-products"
    dataset = PygNodePropPredDataset(name=dataset_name)
    data = dataset[0]

    cluster_data = ld.get_product_clusters()
    data_loader = ld.get_cluster_batches(cluster_data, 4)

    model = models.GCN(data.num_features, args['hidden_dim'],
                dataset.num_classes, args['num_layers'],
                args['dropout']).to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, data_loader, optimizer, device)
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, ')


