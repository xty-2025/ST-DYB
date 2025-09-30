import networkx as nx
import pickle
import numpy as np
import scipy.sparse as sparse

def get_user_graph(adj,user_num): #adj = np.zeros((16, num_nodes, num_nodes), dtype=np.int) 11738users
    # slice_user_graph = np.zeros((user_num,user_num), dtype=int)
    print(' @@@@@@@@@@@@@@@@@@@@@@@@@@get_user_graph start')
    user_adj=np.zeros((user_num,user_num),dtype=int)

    A=adj
    print('adj de bian geshu ',adj.sum()/2)
    A=np.matrix(A)
    A=sparse.coo_matrix(A)
    A=np.dot(A,A).todense()#weights

    for i in range(user_num):
        for j in range(user_num):
            if(A[i,j]>0):
                user_adj[i][j]+=int(A[i,j]/2)
 
#    A=sparse.coo_matrix(A)
#    A=np.dot(A,A).todense()#A*A*A*A weights
#    for i in range(user_num):
#        for j in range(user_num):
#            if(A[i,j]>0):
#                user_adj[i][j]+=A[i,j]/2
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%get_user_graph end')
    return user_adj
    
def get_item_graph(adj,user_num,total_num): #adj = np.zeros((16, num_nodes, num_nodes), dtype=np.int) 11738users
    # slice_user_graph = np.zeros((user_num,user_num), dtype=int)
    print(' @@@@@@@@@@@@@@@@@@@@@@@@@@get_item_graph start')
    item_num=total_num-user_num
    item_adj=np.zeros((item_num,item_num),dtype=int)
    #print(adj.shape)
    A=adj
    print('adj de bian geshu ',adj.sum()/2)
    A=np.matrix(A)
    A=sparse.coo_matrix(A)
    A=np.dot(A,A).todense()#weights

    for i in range(user_num,total_num):
        for j in range(user_num,total_num):
            #print(i,j)
            #print(A.shape)
            if(A[i,j]>0):
                item_adj[i-user_num][j-user_num]+=int(A[i,j]/2)
 
#    A=sparse.coo_matrix(A)
#    A=np.dot(A,A).todense()#A*A*A*A weights
#    for i in range(user_num):
#        for j in range(user_num):
#            if(A[i,j]>0):
#                user_adj[i][j]+=A[i,j]/2
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%get_item_graph end')
    return item_adj


def adj_to_graph(adj):
    print('##########adj_to_graph start')
    newG = nx.MultiGraph()
    for i in range(len(adj)):
        newG.add_node(i)
    print('adj shape:',len(adj))

    for i in range(len(adj)):
        for j in range(len(adj)):
            if(adj[i][j]>0):
                newG.add_edge(i, j,weight=adj[i][j])
    print('edge_num of thie graph:',newG.number_of_edges())
    print('node_num of thie graph:',newG.number_of_nodes())
    return newG
def load_adj(dataset_str,file_name):
    with open("{}/{}".format(dataset_str, file_name), "rb") as f:
        graph = np.load(f)  
    return graph

if __name__ == "__main__":
    s='graphs_yelp.npy'  
    datapath='../beauty'
    G=[]
    total_G=[]
    item_G=[]
    for i in range(1,17):
        s='a'+s
        adjdata_name=s
        dataname=(adjdata_name)
        adj=(load_adj(datapath,dataname))
        print(dataname)
        #total_adj
        total_adj=adj
        total_adj_graph=adj_to_graph(total_adj)
        total_G.append(total_adj_graph)
        #user_adj
        user_adj=get_user_graph(adj,190)#用户个数
        user_adj_graph=adj_to_graph(user_adj)
        G.append(user_adj_graph)#user_adj to user_graph
         #item_adj
        item_adj=get_item_graph(adj,190,800)#用户个数到总个数是项目索引
        item_adj_graph=adj_to_graph(item_adj)
        item_G.append(item_adj_graph)#item_adj to item_graph
    print(len(G))
    print('$$$$$$$$$$$$$$$$$')
    with open('graphs.pkl', 'wb') as f:
       pickle.dump(G, f)
    f.closed
    with open('item_graphs.pkl', 'wb') as f:
       pickle.dump(item_G, f)
    f.closed
    with open('total_graphs.pkl','wb') as f:
        pickle.dump(total_G, f)
    f.closed