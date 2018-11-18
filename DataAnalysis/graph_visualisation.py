import networkx as nx
import matplotlib.pyplot as plt
import numpy

# mst = nx.Graph()
# for coord, weight in minimum_spanning_tree:
#     mst.add_node(coord, weight=weight)

mst_nodes = mst.nodes()
mst_edges = []
for i in range(len(mst_nodes)):
    mst_edges.append([[mst_nodes[i].coords, mst_nodes[i].weight], [mst_nodes[i+1].coords, mst_nodes[i+1].weight])

nx.draw(G)
plt.show()


