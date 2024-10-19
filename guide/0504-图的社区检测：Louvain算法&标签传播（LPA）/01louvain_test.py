import networkx as nx
import matplotlib.pyplot as plt
import community

def simple_graph():
    G = nx.Graph()
    G.name = "Simple Graph"
    node_list = ['A','B','C','D','E','F','G','H','I','J']
    G.add_nodes_from(node_list)
    edge_list  = [('A','B'), ('A','C'),('A','D'), 
              ('B','C'),('B','E'),
              ('C','D'),
              ('D','G'),('D','J'),
              ('E','F'),('E','G'),('E','H'),('E','J'),
              ('F','G'), ('F','J'),
               ('G','J'),
               ('H','I'), ('H','J'),
               ('I','J'),
              ]
    G.add_edges_from(edge_list)
    pos = {'A': [0.1, 0.1], 'B': [0.3, 0.1], 'C': [0.1, 0.4], 'D': [0.3, 0.4],
           'E': [0.5, 0.2], 'F': [0.7, 0.3], 'G': [0.5, 0.4], 
           'H': [0.4, 0.5], 'I': [0.7, 0.5], 'J': [0.7, 0.4]}
    return G, pos


def draw_graph(G, pos,  partition, num):
    plt.figure(num)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=800, 
                           cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show()
    
G,pos = simple_graph()
partition = community.best_partition(G)  # compute communities
draw_graph(G,pos,partition,1)

G = nx.karate_club_graph()
pos = nx.spring_layout(G)  # compute graph layout
partition = community.best_partition(G)  # compute communities
draw_graph(G,pos,partition,2)













