import networkx as nx

G = nx.read_edgelist("../data/facebook_combined.txt", nodetype=int)

local_bridges = nx.local_bridges(G)
set_LB = set(local_bridges)
print("Numero di local bridge nella rete: ", len(set_LB))
print("Numero di bridge nella rete: ", sum(1 for e in set_LB if e[2] == float('inf')))

# Salvataggio edge list del grafo senza i bridge
G_removed = G.copy()
for t in set_LB:
    G_removed.remove_edge(t[0], t[1])

nx.write_edgelist(G_removed, "../data/no_bridge.txt")