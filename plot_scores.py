import matplotlib.pyplot as plt
import pickle
import os


def get_scores_args(folder):
    path = os.path.join('save', folder)
    score_path = os.path.join(path, 'data-scores.pkl')
    arg_path = os.path.join(path, 'args.pkl')
    with open(score_path, 'rb') as f:
        scores = pickle.load(f)
    with open(arg_path, 'rb') as f:
        args = pickle.load(f)
    return scores, args


def get_scores(folder):
    path = os.path.join('save', folder)
    score_path = os.path.join(path, 'data-scores.pkl')
    with open(score_path, 'rb') as f:
        scores = pickle.load(f)
    return scores, None


def plot_train_val_test(scores, name, save_path=None):
    # plt.plot(scores['train'], label='train acc')
    plt.plot(scores['val'], label='val acc')
    plt.plot(scores['test'], label='test acc')
    plt.plot(scores['loss'], label='loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    # plt.title(name, fontsize=16)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_layer_diffs(runs, key, word, title=None, save_path=None):
    for folder in runs.keys():
        scores, _ = get_scores(folder)
        plt.plot(scores[key][:50], label=runs[folder])

    plt.legend(fontsize=12)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(key + " " + word, fontsize=14)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_loss(scores, name, save_path=None):
    plt.plot(scores['loss'], label='loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title(name)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def find_test_val_accs(folder_names):
    pass



if __name__ == "__main__":

    saved_plots_dir = 'plots'
    # Change names of folders to what you have locally
    # folder : title
    # saved_data = {
    #     'ResPostMP_4_layer_128_dim': 'Residual Post Message 4 Layers 128 Hidden Dims Batch Size 4',
    #     'GraphSage_baseline': 'GraphSage with 3 Layers 256 Hidden Dims Batch Size 4',
    #     '6_layer_2': 'Residual Post Message 6 Layers',
    # }
    # Shows accuracy curves
    # for folder in saved_data.keys():
    #     scores, args = get_scores(folder)
    #     # plot_train_val_test(scores, name=saved_data[folder], save_path=None)
    #     print(folder)
    #     print(args)
    #     print("")

    # Plots the test accs for different layer depths
    layer_diffs = {
        '4_layer': '4 layers',
        '5_layer': '5 layers',
        '6_layer_1': '6 layers',
        '10_layer': '10 layers',
    }

    # plot test, val, loss of layer diffs
    plot_layer_diffs(layer_diffs, word='accuracy', key='test', save_path="l_diff_test.png")
    plot_layer_diffs(layer_diffs, word='accuracy', key='val', save_path="l_diff_val.png")
    plot_layer_diffs(layer_diffs, word='', key='loss', save_path="l_diff_loss.png")

    six_layers = {
        '6_layer_1': '6 layers',
        '6_layer_2': '6 layers',
        '6_layer_3': '6 layers',
        '6_layer_4': '6 layers',
        '6_layer_5': '6 layers',
    }
    # plot_layer_diffs(six_layers, title='Acc across runs', key='test', save_path=)

    other_models = {
        'GraphSage_baseline': 'GraphSage with Linear Head',
        'no_mp_4': '4 Layer Residual GCN with Linear Head',
        '6_layer_5': '6 layer Residual GCN with ResNet Head',
    }

    plot_layer_diffs(other_models, word='accuracy', key='test', save_path="other_models_test.png")
    plot_layer_diffs(other_models, word='accuracy', key='val', save_path="other_models_test.png")
    plot_layer_diffs(other_models, word='', key='loss', save_path="other_loss.png")

    sum_vs_mean = {
        '6_layer_1': 'Mean Aggregation',
        'postMP_sum_agg': 'Sum Aggregation',
    }
    plot_layer_diffs(other_models, word='accuracy', key='test', save_path="sum_vs_mean.png")

    # Get max test_valid

