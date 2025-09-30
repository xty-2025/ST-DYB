import argparse
import pickle as pkl
from cellutil import array_to_matrix,load_adj
from preprocess import get_context_pairs
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--total_graph_name', type=str, nargs='?',
                        default='/root/DySAT_pytorch10_27copy/data/beauty/total_graphs.pkl',
                        help='total graph name')
    parser.add_argument('--context_pairs_path', type=str, nargs='?',
                        default='/root/DySAT_pytorch10_27copy/data/beauty/context_pairs.pkl',
                        help='The path of context_pairs')
    parser.add_argument('--datapath', type=str, nargs='?', default='/root/DySAT_pytorch10_27copy/data/beauty',
                        help='datapath')
    args = parser.parse_args()
    with open((args.total_graph_name), "rb") as f:
        total_graphs= pkl.load(f)
    total_adjs_arr = load_adj(args.datapath)
    total_adjs = array_to_matrix(total_adjs_arr)
    print(len(total_adjs))
    context_pairs_train = get_context_pairs(total_graphs, total_adjs)
    with open(args.context_pairs_path, 'wb') as f:
        pkl.dump(context_pairs_train, f)
