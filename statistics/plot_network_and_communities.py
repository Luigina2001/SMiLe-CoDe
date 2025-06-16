import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # noqa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

G = nx.read_edgelist("./data/facebook_combined.txt", nodetype=int)

# Plot dell'intero grafo
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=20, alpha=0.7)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.axis('off')
plt.title("Grafo Facebook completo")
plt.savefig("./statistics/output/netowork.png", dpi=150)
plt.close()

# Rilevazione delle comunità (Louvain) e plot
partition = community_louvain.best_partition(G)  # dict: nodo → id_comunità

comms = list(set(partition.values()))
mapping = {comm: idx for idx, comm in enumerate(comms)}
node_colors = [mapping[partition[v]] for v in G.nodes()]

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos,
                       node_size=20,
                       node_color=node_colors,
                       cmap=plt.cm.tab20,  # noqa
                       alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.2)
plt.axis('off')
plt.title("Comunità (Louvain) nel grafo Facebook")
plt.savefig("./statistics/output/communities.png", dpi=150)
plt.close()
