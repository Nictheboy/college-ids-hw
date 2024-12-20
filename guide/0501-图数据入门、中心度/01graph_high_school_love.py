import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def draw(G, pos, measures, measure_name,num):
    
    plt.figure(num)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()
    
def school_dating_graph():
    students = set(range(11))
    G = nx.Graph()
    G.name = "Simple Dating Graph"
    G.add_nodes_from(students)
    dating_rel = [(0,3), (1,3), (2,3), (3,4), (4,5), (4,9), 
                   (5,6), (6,7), (6,8), (6,9), (9,10)]
    G.add_edges_from(dating_rel)
    # You may want to try automatic layout
    #pos = nx.spring_layout();
    pos = {0: [0.1, 0.6], 1: [0.1, 0.5], 2: [0.1, 0.4], 3: [0.2, 0.5],
           4: [0.3, 0.5], 5: [0.45, 0.7], 6: [0.6, 0.5], 7: [0.7, 0.6],
           8: [0.7, 0.4], 9: [0.45, 0.3], 10: [0.45, 0.2]}
    return G, pos
def school_dating_digraph():
    students = set(range(11))
    G = nx.DiGraph()
    G.name = "Simple Dating Graph"
    G.add_nodes_from(students)
    dating_rel = [(0,3), (1,3), (2,3), (3,4), (4,5), (4,9), 
                   (5,6), (6,7), (6,8), (6,9), (9,10)]
    G.add_edges_from(dating_rel)
    # You may want to try automatic layout
    #pos = nx.spring_layout();
    pos = {0: [0.1, 0.6], 1: [0.1, 0.5], 2: [0.1, 0.4], 3: [0.2, 0.5],
           4: [0.3, 0.5], 5: [0.45, 0.7], 6: [0.6, 0.5], 7: [0.7, 0.6],
           8: [0.7, 0.4], 9: [0.45, 0.3], 10: [0.45, 0.2]}
    return G, pos

G,pos = school_dating_graph()
draw(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality',1)
print("betweenness_centrality",nx.betweenness_centrality(G))

draw(G, pos, nx.degree_centrality(G), 'Degree Centrality',2)
print("degree_centrality",nx.degree_centrality(G))

draw(G, pos, nx.closeness_centrality(G), 'Closeness Centrality',3)
print("closeness_centrality",nx.closeness_centrality(G))



pair_list = [
    [0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],[0,10],
    [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],
    [2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],[2,10],
    [3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
    [4,5],[4,6],[4,7],[4,8],[4,9],[4,10],
    [5,6],[5,7],[5,8],[5,9],[5,10],
    [6,7],[6,8],[6,9],[6,10],
    [7,8],[7,9],[7,10],
    [8,9],[8,10],
    [9,10]
    ]
for pair in pair_list:
    first = pair[0]
    second = pair[1]
    print(first, second, ":",[p for p in nx.all_shortest_paths(G,source=first,target=second)])



G,pos = school_dating_digraph()
draw(G, pos, nx.pagerank(G, alpha=0.85), 'DiGraph PageRank',4)
print("pagerank",nx.pagerank(G))



DiG = nx.DiGraph()
DiG.add_edges_from([(2, 3), (3, 2), (4, 1), (4, 2), (5, 2), (5, 4),
                    (5, 6), (6, 2), (6, 5), (7, 2), (7, 5), (8, 2),
                    (8, 5), (9, 2), (9, 5), (10, 5), (11, 5)])
dpos = {1: [0.1, 0.9], 2: [0.4, 0.8], 3: [0.8, 0.9], 4: [0.15, 0.55],
        5: [0.5,  0.5], 6: [0.8,  0.5], 7: [0.22, 0.3], 8: [0.30, 0.27],
        9: [0.38, 0.24], 10: [0.7,  0.3], 11: [0.75, 0.35]}

h,a = nx.hits(DiG)
draw(DiG, dpos, h, 'DiGraph HITS Hubs',5)
draw(DiG, dpos, a, 'DiGraph HITS Authorities',6)
