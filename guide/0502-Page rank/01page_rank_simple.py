import numpy as np
import networkx as nx

def pagerank_naive (DiG, pinit, max_iter=36):
    # Adjacency Matrix
    A = nx.to_numpy_matrix(DiG)
    # Out-Degree -> M -> M^{-1}
    D = np.sum(A,axis=1)
    M = np.diag(D.A1)
    M_I = np.linalg.inv(M)
    
    L = M_I @ A # Must use Python3 to use @
    p = pinit
    
    for i in range(max_iter):
        p = p @ L
        print (p)

DiG = nx.DiGraph()
DiG.name = "Simple Graph"
node_list = ['A','B','C','D']
DiG.add_nodes_from(node_list)
edge_list  = [('A','B'), ('B','C'), ('B','D'), ('C','D'), ('D','A')
              ]
DiG.add_edges_from(edge_list)

n= len(DiG)
p = np.ones(n)/n#(0.25,0.25,0.25,0.25)
print(p, ":")
pagerank_naive(DiG,p)
print("")

p = [1,0,0,0]
p = np.asarray(p)
print(p, ":")
pagerank_naive(DiG,p)
print("")

p = [0,1,0,0]
p = np.asarray(p)
print(p, ":")
pagerank_naive(DiG,p)
print("")
