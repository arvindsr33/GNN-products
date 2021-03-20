import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import load_data as ld
import models
from tqdm import tqdm
import time
import collections
import pickle
import copy
from datetime import datetime
import os

@torch.enable_grad()
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
            out = model(batch.x, batch.edge_index)[batch.train_mask]
            y = torch.flatten(batch.y[batch.train_mask])
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_nodes += nodes
            progress_bar.update(1)

    return total_loss / total_nodes


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    print("starting")
    print(data)

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
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def test_cluster(model, data_loader, evaluator, device):
    model.eval()
    train_preds = [] 
    val_preds = []
    test_preds = []

    train_labels = []
    val_labels = []
    test_labels = []

    # get the preds of each cluster
    with tqdm(total=len(data_loader)) as progress_bar:
        for batch in data_loader:
            batch.to(device)
            out = model(batch.x, batch.edge_index)

            if batch.train_mask.sum() != 0:
                train_pred = out[batch.train_mask].argmax(dim=1, keepdim=True)
                train_label = batch.y[batch.train_mask]
                train_preds.append(train_pred.cpu())
                train_labels.append(train_label.cpu())

            if batch.valid_mask.sum() != 0:
                val_pred = out[batch.valid_mask].argmax(dim=1, keepdim=True)
                val_label = batch.y[batch.valid_mask]
                val_preds.append(val_pred.cpu())
                val_labels.append(val_label.cpu())

            if batch.test_mask.sum() != 0:
                test_pred = out[batch.test_mask].argmax(dim=1, keepdim=True)
                test_label = batch.y[batch.test_mask]
                test_preds.append(test_pred.cpu())
                test_labels.append(test_label.cpu())

            progress_bar.update(1)

    train_label = torch.cat(train_labels, dim=0)
    train_pred = torch.cat(train_preds, dim=0)
    val_label = torch.cat(val_labels, dim=0)
    val_pred = torch.cat(val_preds, dim=0)
    test_label = torch.cat(test_labels, dim=0)
    test_pred = torch.cat(test_preds, dim=0)

    train_acc = evaluator.eval({
        'y_true': train_label,
        'y_pred': train_pred
    })['acc']
    val_acc = evaluator.eval({
        'y_true': val_label,
        'y_pred': val_pred
    })['acc']
    test_acc = evaluator.eval({
        'y_true': test_label,
        'y_pred': test_pred
    })['acc']

    return train_acc, val_acc, test_acc


def print_scores(epoch, loss, train_acc, valid_acc, test_acc):
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')


def save_data(obj, path, name):
    file_name = f"data-{name}.pkl"
    file_name = os.path.join(path, file_name)
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_model(model, args, save_dir):
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    args_path = os.path.join(save_dir, 'args.pkl')
    with open(args_path, 'wb') as f:
        pickle.dump(args, f)


def load_model(save_dir):
    args_path = os.path.join(save_dir, 'args.pkl')
    model_path = os.path.join(save_dir, 'model.pt')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    model = models.get_model(args)
    model.load_state_dict(torch.load(model_path))
    return args, model


if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    args = {
        'device': device,
        'num_layers': 4,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.01,
        'epochs': 50,
        'return_embeds': False,
        # 'model_type': 'GraphSage',
        # 'model_type': 'GCN',
        # 'model_type': 'ResPostMP',
        'model_type': 'ResidualMP',
        'heads': 1,
        'batch_size': 4,
        'post_hidden': 256,
        'message_hidden': 256,
        'normalize': False,
    }
    dataset_name = "ogbn-products"

    save_time = datetime.now().strftime("%m-%d_%H_%M_%S")
    save_dir = os.path.join("save", args['model_type'] + "_" + save_time)
    os.makedirs(save_dir)
    print("SAVE PATH:", save_dir)

    cluster_data, dataset, data, split_idx = ld.get_product_clusters()
    data_loader = ld.get_cluster_batches(cluster_data, args['batch_size'])
    evaluator = Evaluator(name=dataset_name)
    args['input_dim'] = data.num_features
    args['output_dim'] = dataset.num_classes

    model = models.get_model(args)
    print(model)

    # model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss
    model.to(device)

    scores = {'loss': [], 'train': [], 'val': [], 'test': []}
    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, data_loader, optimizer, device)
        result = test_cluster(model, data_loader, evaluator, device)
        train_acc, valid_acc, test_acc = result
        print_scores(epoch, loss, train_acc, valid_acc, test_acc)

        # Save loss and accuracies to data dictionary
        scores["loss"].append(loss)
        scores["train"].append(train_acc)
        scores["val"].append(valid_acc)
        scores["test"].append(test_acc)

        # copy model params
        if valid_acc > best_valid_acc:
            best_model = copy.deepcopy(model)

    save_model(model, args, save_dir)
    save_data(scores, save_dir, "scores")

    # save_dir = os.path.join('save', 'GCN_03-19_22_05_24')
    # prev_args, model = load_model(save_dir)
    # model.to(device)
    # result = test_cluster(model, data_loader, evaluator, device)
    # print(result)



