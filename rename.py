import plot_scores
import sys
import os
import shutil

if __name__ == '__main__':
    dirname = sys.argv[1]
    output_folder = None
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]

    for d in os.listdir(dirname):
        orig_name = d[0:d.find('_')]
        run_at = d[d.find('_')+1:]
        scores, args = plot_scores.get_scores(d)
        new_name = "{0}_{1}_layer_{2}_dim_{3}_batch_{4}".format(
           args['model_type'], args['num_layers'], args['hidden_dim'], args['batch_size'], run_at)
        if new_name != d:
            print("Renaming: ", d, "", new_name)
        print(new_name, args)
        if output_folder:
            shutil.copytree(os.path.join(dirname, d), os.path.join(output_folder, new_name))

