import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict


def array_to_matrix(adjs):
    f_adj = []
    for adj in adjs:
        a = sp.csr_matrix(adj)
        f_adj.append(a)
    return f_adj
def load_adj(dataset_str):
    adjs=[]
    s = 'graphs_yelp.npy'
    for i in range(1,17):
        s='a'+s
        with open("{}/{}".format(dataset_str,s), "rb") as f:
            adj = np.load(f)
        adjs.append(adj)

    return adjs
