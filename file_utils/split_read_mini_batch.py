import os
import sys

def split_file_data(file_path, num_lines_per_file):
    f = open(file_path)
    rows = []
    count = 0
    batch_size = 50000
    batch_id = 0
    for row in f:
        rows.append(row)
        count += 1
        if count == batch_size:
            n_path, n_ext = os.path.splitext(file_path)
            w = open(n_path + '-' + str(batch_id).rjust(5, '0') + n_ext, 'w')
            for r in rows:
                w.write(r)
            w.close()
            rows = []
            count = 0
            batch_id += 1


if __name__ == '__main__':
  data_dir_name = sys.argv[1]
  files = ["edge.csv", "node-feat.csv", "node-label.csv"]
  for file_name in files:
     file_path = os.path.join(data_dir_name, file_name)
     split_file_data(file_path, 50000)
