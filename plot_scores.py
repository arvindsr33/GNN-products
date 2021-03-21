import matplotlib.pyplot as plt
import pickle
import os


def get_scores(folder):
    path = os.path.join('save', folder)
    score_path = os.path.join(path, 'data-scores.pkl')
    arg_path = os.path.join(path, 'args.pkl')
    with open(score_path, 'rb') as f:
        scores = pickle.load(f)
    with open(arg_path, 'rb') as f:
        args = pickle.load(f)
    return scores, args


def plot_train_val_test(scores, name, save_path=None):
    plt.plot(scores['train'], label='train acc')
    plt.plot(scores['val'], label='val acc')
    plt.plot(scores['test'], label='test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(name)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_layer_diffs(runs, title, key, save_path=None):
    for folder in runs.keys():
        scores, _ = get_scores(folder)
        plt.plot(scores[key], label=runs[folder])

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(key + " accuracy")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_loss(scores, name, save_path=None):
    plt.plot(scores['loss'], label='loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(name)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":

    # Change names of folders to what you have locally
    # folder : title
    saved_data = {
        'ResPostMP_4_layer_128_dim': 'Residual Post Message 4 Layers 128 Hidden Dims Batch Size 4',
        'GraphSage_baseline': 'GraphSage with 3 Layers 256 Hidden Dims Batch Size 4',
        'ResPostMP_6_layer_128_dim_32_batch': 'Residual Post Message 6 Layers'
    }

    # Shows accuracy curves
    for folder in saved_data.keys():
        scores, args = get_scores(folder)
        plot_train_val_test(scores, name=saved_data[folder], save_path=None)
        print(args)
        print("")

    # Plots the test accs for different layer depths
    layer_diffs = {
        'ResPostMP_4_layer_128_dim': '4 Layers Batch 4',
        'ResPostMP_6_layer_128_dim_32_batch': '6 layers Batch 32'
    }
    plot_layer_diffs(layer_diffs, title='Test Accuracy vs Layer Depth', key='test', save_path=None)

    # Add any extra combinations of folders to plot on the same graph if needed
