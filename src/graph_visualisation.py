import networkx as nx
import matplotlib.pyplot as plt
import numpy
from .load_data import jobs_centres, emp_edu_centres, result

mst = nx.Graph()

class Nodes:
    __init__()

for coord in jobs_centres:
    mst.add_node(coord, value=value)

for num_row, row in enumerate(result.itercols()):
    for num_col, dist in enumerate(row):
        mst.add_edge(num_row, num_col, length=dist)

mst_edges = []
for i in range(len(mst.nodes()) - 1):
    mst_edges.append([[mst.nodes[i].coords, mst.nodes[i].value], [mst.nodes[i+1].coords, mst.nodes[i+1].value]])

nx.draw(mst)
plt.show()
