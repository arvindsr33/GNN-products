import random


def sample_edges(edge_file_name,output_file_name):
    d1 = {}
    l = []
    for i in range(1000):
        d1[random.randint(0, 61859140)] = 1

    f = open(edge_file_name)
    c = 0
    out =[]
    while True:
        l = f.readline()
        if len(l) == 0:
            break
        if c in d1:
            out.append(l)
        c += 1

    w = open(output_file_name, 'w')
    for line in out:
        w.write(line)

    w.close()


def get_node_features(node_file, node_label_file, edge_file, node_feature_output, node_label_output):
    f = open(edge_file)
    nodes = []
    for line in f.readlines():
        values = line.strip().split(',')
        for x in values:
            if len(x.strip()) > 0:
                nodes.append(int(x))

    print("Num nodes before dedup=", len(nodes))
    nodes = set(nodes)
    print("Num nodes=", len(nodes))

    c = 0
    nf = open(node_file)
    nl = open(node_label_file)
    outf = []
    outl = []
    while True:
        features = nf.readline()
        label = nl.readline()

        if not features:
            break
        if c in nodes:
            outf.append(features)
            outl.append(label)
        c = c + 1

    w = open(node_feature_output, 'w')
    for line in outf:
        w.write(line)
    w.close()

    w = open(node_label_output, 'w')
    for line in outl:
        w.write(line)
    w.close()


if __name__ == '__main__':
    data_dir = "C:\\Users\\shaon\\Desktop\\CS224W\\CS224W_2021\\CS224W_PROJECT\\dataset\\ANALYZE\\ogbn_products\\raw"

    edge_input_file =  data_dir + "\\edge.csv"
    edge_output_file = data_dir + "\\out\\output.csv"

    sample_edges(edge_input_file, edge_output_file)

    node_file = data_dir + "\\node-feat.csv"
    node_label_file = data_dir + "\\node-label.csv"
    node_feature_output = data_dir + "\\out\\node-feat.csv"
    node_label_output = data_dir + "\\out\\node-label.csv"
    get_node_features(node_file, node_label_file, edge_output_file, node_feature_output, node_label_output)







