import random
def sample_edges(edge_file_name,output_file_name):
    d1 = {}
    l = []
    for i in range(1000):
        d1[random.randint(0,61859140)] = 1

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
        w.write('\n')

    w.close()

if __name__ == '__main__':
    sample_edges("C:\\Users\\shaon\\Desktop\\CS224W\\CS224W_2021\\CS224W_PROJECT\\dataset\\ANALYZE\\ogbn_products\\raw\\edge.csv","C:\\Users\\shaon\\Desktop\\CS224W\\CS224W_2021\\CS224W_PROJECT\\dataset\\ANALYZE\\ogbn_products\\raw\\out\\output.csv")







