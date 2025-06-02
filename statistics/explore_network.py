import snap # noqa


# Caricamento del grafo
G = snap.LoadEdgeList(snap.PUNGraph, "../data/facebook_combined.txt", 0, 1, ' ') # noqa


# 1. ============= Network Statistics =============
print(" ============= Network Statistics =============")
print(f"Nodes: {G.GetNodes()}")
print(f"Edges: {G.GetEdges()}")

# Largest WCC e SCC (per grafi non orientati coincidono)
MxWcc = G.GetMxWcc()
print(f"\nNodes in largest WCC: {MxWcc.GetNodes()} ({MxWcc.GetNodes()/G.GetNodes():.3f})")
print(f"Edges in largest WCC: {MxWcc.GetEdges()} ({MxWcc.GetEdges()/G.GetEdges():.3f})")

MxScc = G.GetMxScc()
print(f"Nodes in largest SCC: {MxScc.GetNodes()} ({MxScc.GetNodes()/G.GetNodes():.3f})")
print(f"Edges in largest SCC: {MxScc.GetEdges()} ({MxScc.GetEdges()/G.GetEdges():.3f})")

# Clustering Coefficient
clust_coeff = G.GetClustCf()
print(f"\nAverage clustering coefficient: {clust_coeff:.4f}")

# Triangles
triangles = G.GetTriads()
print(f"Number of triangles: {triangles}")

# Fraction of closed triangles (Global Clustering Coefficient)
sum_deg_choose2 = sum(NI.GetDeg()*(NI.GetDeg()-1)//2 for NI in G.Nodes())
fraction_closed = (3*triangles)/sum_deg_choose2 if sum_deg_choose2 > 0 else 0
print(f"Fraction of closed triangles: {fraction_closed:.6f}")

# Diameter (approssimato)
diam = G.GetBfsFullDiam(96, False)
print(f"\nDiameter (longest shortest path): {diam}")

# Effective Diameter
eff_diam = G.GetAnfEffDiam(False, 0.9, 96)
print(f"90-percentile effective diameter: {eff_diam:.1f}")


# 2. ============= Community Statistics (if the community file is already specified) =============
'''print("\n ============= Community Statistics =============")
community_sizes = []
membership_counts = snap.TIntH()  # noqa node -> number of communities in which it appears

with open("../data/comunita.txt", "r") as f:
    for line in f:
        nodes = list(map(int, line.strip().split()))
        community_sizes.append(len(nodes))
        for node in nodes:
            if not membership_counts.IsKey(node):
                membership_counts.AddDat(node, 0)
            membership_counts.AddDat(node, membership_counts.GetDat(node) + 1)

num_communities = len(community_sizes)
avg_community_size = sum(community_sizes)/num_communities if num_communities > 0 else 0

total_memberships = 0
it = membership_counts.BegI()
while not it.IsEnd():
    total_memberships += it.GetDat()
    it.Next()

avg_membership = total_memberships/G.GetNodes() if G.GetNodes() > 0 else 0

print(f"Number of communities: {num_communities}")
print(f"Average community size: {avg_community_size:.2f}")
print(f"Average membership size: {avg_membership:.2f}")'''


# 3. ============= Community Statistics (if community file is not specified) =============

# Community Detection (Clauset-Newman-Moore)
CmtyV = snap.TCnComV()  # noqa
modularity = snap.CommunityCNM(G, CmtyV)  # noqa

print("\n ============= Community Statistics =============")
community_sizes = []
membership_counts = snap.TIntH()  # noqa node -> number of communities in which it appears

for community in CmtyV:
    nodes = list(community)
    community_sizes.append(len(nodes))
    for node in nodes:
        if not membership_counts.IsKey(node):
            membership_counts.AddDat(node, 0)
        membership_counts.AddDat(node, membership_counts.GetDat(node) + 1)


num_communities = len(community_sizes)
avg_community_size = sum(community_sizes) / num_communities if num_communities > 0 else 0

total_memberships = 0
it = membership_counts.BegI()
while not it.IsEnd():
    total_memberships += it.GetDat()
    it.Next()

avg_membership = total_memberships / G.GetNodes() if G.GetNodes() > 0 else 0


print(f"Number of communities: {num_communities}")
print(f"Average community size: {avg_community_size:.2f}")
print(f"Average membership size (nodes per community): {avg_membership:.2f}")
print(f"Modularity: {modularity:.4f}")
