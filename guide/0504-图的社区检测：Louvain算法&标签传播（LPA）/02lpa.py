#https://www.programmersought.com/article/19431905738/

import networkx as nx

G=nx.karate_club_graph()

from networkx.algorithms import community
def label_propagation_community(G):
    communities_generator = list(community.label_propagation_communities(G))
    m = []
    for i in communities_generator:
        m.append(list(i))
    return m

g=label_propagation_community(G)
print(g)

map_all ={}
community_num = 0;
for list_one in g:
    for list_item in list_one:
        map_all[list_item] = community_num
    community_num = community_num+1

print( map_all)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

partition =map_all
# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=160,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
