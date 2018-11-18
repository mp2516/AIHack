### MINIMUM SPANNING TREE

from collections import deque, defaultdict
import numpy as np
from math import cos, asin, sqrt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from heapq import heapify, heappush, heappop
from weights import find_flows

def calculate_distance(location_1, location_2):
    lat1 = location_1[1]
    lon1 = location_1[0]
    lat2 = location_2[1]
    lon2 = location_2[0]
    p = 0.017453292519943295  # Pi/180
    a = np.sin(p*(lat1-lat2)/2)**2 + np.cos(p*lat1)*np.cos(p*lat2)*(np.sin(p*(lon1 - lon2)/2)**2)
    return 12742 * asin(sqrt(a))


def calculate_all_distances(Jobs, Workers,all_weights):
    all_nodes = np.concatenate((Jobs, Workers))
    dimension = len(all_nodes)

    all_coords = np.zeros((dimension, 3))
    for i in range(0, dimension):
        all_coords[i, 0] = all_nodes[i, 0]
        all_coords[i, 1] = all_nodes[i, 1]
        all_coords[i, 2] = all_weights[i]  # all_nodes[i,2] #This needs to go back in

    all_distances = np.zeros((dimension, dimension))
    for i in range(0, dimension):
        for j in range(i + 1, dimension):
            all_distances[i][j] = calculate_distance(all_nodes[i], all_nodes[j])

    return all_distances, all_coords


# Mimimum Spanning Tree

def min_span_tree(Jobs, Workers, job_weights, worker_weights):
    all_weights = np.concatenate((job_weights, worker_weights))
    all_distances, all_coords = calculate_all_distances(Jobs, Workers,all_weights)
    Tcsr = minimum_spanning_tree(all_distances)
    return Tcsr.toarray().astype(int), all_coords


######### FLOW NETWORK: FORD FOLKERSON

"""

inf = float('inf')


def residual(G, x):
    '''This function computes the residual network'''
    Gf = []
    for u in range(len(G)):
        Gf.append(dict())
    for u in range(len(G)):
        for v in G[u]:
            if u != v:
                cap = G[u][v]
                if cap - x[u, v]: Gf[u].update({v: cap - x[u, v]})
                if x[u, v]: Gf[v].update({u: x[u, v]})
    return Gf

def aug_paths(G, s, t, x):  # Walk the graph from node s, x is a flow
    '''This function determines if there is an augmenting path betweeen
       s and t using BFS; if there exists one, both the traversal tree and
       the residual capacity are returned'''
    P, Q, F = dict(), deque(), dict()  # Predecessors, "to do" queue, flow label
    Gf = residual(G, x)  # Compute residual network
    Q.append((s, None))  # We plan on starting with s, no predecessor
    while Q:  # Still nodes to visit
        u = Q.popleft()  # Pick first in, constant time operation
        if u[0] == t:  # Reached t? Augmenting path found
            P[u[0]] = u[1]  # Update traversal tree
            F[u[0]] = min(F[u[1]], Gf[u[1]][u[0]])  # Update flow label, min between flow predecessor and current edge
            return P, F[u[0]]  # Return traversal tree and residual capacity
        if u[0] in P: continue  # Already visited? Skip it
        P[u[0]] = u[1]  # We've visited it now, predecessor is u[1]
        try:
            F[u[0]] = min(F[u[1]], Gf[u[1]][u[0]])  # Update flow label
        except:
            F[u[0]] = inf  # u[0] = s?, flow transported to the predessor is inf (no predecessor)
        for v in Gf[u[0]]:
            Q.append((v, u[0]))  # Schedule all neighbors, predecessor is u[0]
    return None, 0  # No augmenting path found


def ford_fulkerson(G, s, t, aug=aug_paths):  # Max flow from s to t
    f = defaultdict(int)  # Transpose and flow
    while True:  # While we can improve things
        P, c = aug_paths(G, s, t, f)  # Aug. path and capacity/slack
        if c == 0: return f  # No augm. path found? Done!
        u = t  # Start augmentation
        while u != s:  # Backtrack to s
            u, v = P[u], u  # Shift one step
            if v in G[u]:
                f[u, v] += c  # Forward edge? Add slack
            else:
                f[v, u] -= c  # Backward edge? Cancel slack


# Example 1
# Flow network
one, two, three, four = range(4)
g1 = [
    {two: 2, three: 4},  # ONE
    {three: 3, four: 1},  # TWO
    {four: 5},  # THREE
    {}  # FOUR
]

# Computing maximum flow
res_flow = ford_fulkerson(g1, one, four)
print(res_flow)
print("\n")

"""
