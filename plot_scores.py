import matplotlib.pyplot as plt
import pickle
import os


def get_scores(folder):
    path = os.path.join('save_processed', folder)
    score_path = os.path.join(path, 'data-scores.pkl')
    arg_path = os.path.join(path, 'args.pkl')
    with open(score_path, 'rb') as f:
        scores = pickle.load(f)
    with open(arg_path, 'rb') as f:
        args = pickle.load(f)
    return scores, args


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


def plot_layer_diffs(runs, title, key, save_path=None):
    for folder in runs.keys():
        scores, _ = get_scores(folder)
        plt.plot(scores[key], label=runs[folder])

    plt.legend(fontsize=12)
    # plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(key + " accuracy", fontsize=14)
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


if __name__ == "__main__":

    saved_plots_dir = 'plots'
    # Change names of folders to what you have locally
    # folder : title
    saved_data = {
        # 'ResPostMP_4_layer_128_dim': 'Residual Post Message 4 Layers 128 Hidden Dims Batch Size 4',
        # 'GraphSage_baseline': 'GraphSage with 3 Layers 256 Hidden Dims Batch Size 4',
        # 'ResPostMP_6_layer_128_dim_32_batch': 'Residual Post Message 6 Layers',
        'ResPostMP_10_layer_128_dim_32_batch_03-21_01_59_14':
            'Residual Post Message 10 Layers 128 Hidden Dims',
        'ResPostMP_3_layer_128_dim_32_batch_03-20_22_39_47':
            'Residual Post Message 3 Layers 128 Hidden Dims',
        'ResPostMP_3_layer_128_dim_32_batch_03-20_23_33_57':
            'Residual Post Message 3 Layers 128 Hidden Dims',
        'ResPostMP_3_layer_128_dim_32_batch_03-20_23_37_53':
            'Residual Post Message 3 Layers 128 Hidden Dims',
        'ResPostMP_3_layer_256_dim_32_batch_03-20_15_19_55':
            'Residual Post Message 3 Layers 256 Hidden Dims',
        'ResPostNoMP_4_layer_128_dim_32_batch_03-21_03_37_41':
            'Linear Post Message 4 Layers 128 Hidden Dims',
        'ResPostMP_5_layer_128_dim_32_batch_03-21_01_37_45':
            'Residual Post Message 5 Layers 128 Hidden Dims',
        'ResPostMP_6_layer_128_dim_32_batch_03-21_00_46_06':
            'Residual Post Message 6 Layers 128 Hidden Dims',
        'ResPostMP_6_layer_128_dim_32_batch_03-21_04_15_24':
            'Residual Post Message 6 Layers 128 Hidden Dims',
        'ResPostMP_6_layer_128_dim_32_batch_03-21_04_15_26':
            'Residual Post Message 6 Layers 128 Hidden Dims',
        'ResPostMP_6_layer_128_dim_32_batch_03-21_05_33_56':
            'Residual Post Message 6 Layers 128 Hidden Dims',
        'ResPostNoMP_7_layer_128_dim_32_batch_03-21_03_40_49':
            'Linear Post Message 7 Layers 128 Hidden Dims',
        'ResidualMP_3_layer_256_dim_32_batch_03-20_15_17_25':
            'Residual Post Message 3 Layers 256 Hidden Dims',
    }

    # Shows accuracy curves
    for folder in saved_data.keys():
        scores, args = get_scores(folder)
        save_path = os.path.join(saved_plots_dir, "scores_{0}.jpg".format(folder))
        plot_train_val_test(scores, name=saved_data[folder], save_path=save_path)
        print(args)
        print("")

    # Plots the test accs for different layer depths
    layer_diffs = {
        # 'ResPostMP_4_layer_128_dim': '4 layers Batch 4',
        # 'ResPostMP_6_layer_128_dim_32_batch': '6 layers Batch 32',
        'ResPostNoMP_4_layer_128_dim_32_batch_03-21_03_37_41': '4 layers Batch 32',
        'ResPostMP_6_layer_128_dim_32_batch_03-21_00_46_06': '6 layers Batch 32',
        'ResPostMP_5_layer_128_dim_32_batch_03-21_01_37_45': '5 layers Batch 32',
    }
    save_path = os.path.join(saved_plots_dir, "layers.jpg")
    plot_layer_diffs(layer_diffs, title='Test Accuracy vs Layer Depth', key='test', save_path=save_path)

    # Add any extra combinations of folders to plot on the same graph if needed
