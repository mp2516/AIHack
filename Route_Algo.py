### MINIMUM SPANNING TREE

from collections import deque, defaultdict
import numpy as np
from math import cos, asin, sqrt

Jobs = [
    {12.34, 34.65, 0.3},
    {34.57, 23.76, 0.4},
    {23.56, 29.86, 0.3}]

Workers = [
    {12.34, 32.65, 0.1},
    {39.57, 21.76, 0.2},
    {37.57, 26.76, 0.4},
    {34.57, 27.76, 0.4},
    {21.56, 32.86, 0.3}]

from heapq import heapify, heappush, heappop


def calculate_distance(location_1, location_2):
    lat1 = location_1[0]
    lon1 = location_1[1]
    lat2 = location_2[0]
    lon2 = location_2[1]
    p = 0.017453292519943295  # Pi/180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))
    # return np.sqrt((location_1[0] - location_2[0]) ** 2 + (location_1[1] - location_2[1]) ** 2)


def calculate_all_distances(Jobs, Workers):
    all_nodes = Jobs + Workers
    dimension = len(all_nodes)
    all_distances = np.zeros((dimension, dimension))
    for i in range(0, dimension):
        for j in range(0, dimension):
            all_distances[i][j] = calculate_distance(all_nodes(i), all_nodes(j))


# Mimimum Spanning Tree

def kruskal_heap(G):
    T = set()  # MST as a set of edges
    subgraphs = {u: {u} for u in range(len(G))}  # Initialize subgraphs
    E = [(G[u][v], u, v) for u in range(len(G)) for v in G[u]]  # List of edges with weight
    heapify(E)
    while len(T) < len(G) - 1:
        _, u, v = heappop(E)  # Smallest edge
        if subgraphs[u].intersection(subgraphs[v]) == set():  # Smallest edge is not in subgraphs
            T.add((u, v))
            subgraphs[u] = subgraphs[u].union(subgraphs[v])
            subgraphs[v] = subgraphs[v].union(subgraphs[u])
    return T


bos, dfw, jfk, lax, mia, ordd, sfo, bwi, pvd = range(9)
g1 = [
    {sfo: 2704, ordd: 867, jfk: 187, mia: 1258},  # BOS
    {sfo: 1464, lax: 1235, ordd: 802, mia: 1121},  # DFW
    {ordd: 740, mia: 1090},  # JFK
    {sfo: 337, mia: 2342},  # LAX
    {},  # MIA
    {sfo: 1846},  # ORD
    {}  # SFO
]

print(kruskal_heap(g2))

######### FLOW NETWORK: FORD FOLKERSON

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
