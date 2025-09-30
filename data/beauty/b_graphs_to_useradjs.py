import numpy as np

from matplotlib import pyplot as plt
import pickle as pkl
import networkx as nx
with open('graphs.pkl', "rb") as f:
    graphs = pkl.load(f)
    # nx.draw_networkx(graphs[15])
    # plt.show()
print('sucessful1')
print(type(graphs[0]))
adjs = [nx.adjacency_matrix(g) for g in graphs]
print('successful2')
with open('user_adj.pkl', 'wb') as f:
    pkl.dump(adjs, f)
    
with open('item_graphs.pkl', "rb") as f:
    graphs = pkl.load(f)
adjs = [nx.adjacency_matrix(g) for g in graphs]
with open('item_adj.pkl', 'wb') as f:
    pkl.dump(adjs, f)
print('successful3')