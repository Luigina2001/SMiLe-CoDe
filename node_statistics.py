import networkx as nx
import matplotlib.pyplot as plt


def calcola_statistiche(G):  # noqa
    # 1. Degree
    degree_dict = dict(G.degree())
    nx.set_node_attributes(G, degree_dict, name="degree")

    # 2. Degree centrality
    deg_centrality = nx.degree_centrality(G)
    nx.set_node_attributes(G, deg_centrality, name="deg_centrality")

    # 3. Betweenness centrality
    betw = nx.betweenness_centrality(G, normalized=True)
    nx.set_node_attributes(G, betw, name="betweenness")

    # 4. Clustering coefficient
    clust = nx.clustering(G)
    nx.set_node_attributes(G, clust, name="clustering")

    return degree_dict, deg_centrality, betw, clust


def salva_istatogramma_gradi(degree_dict, nome_file="./statistiche/istogramma_gradi.png"):
    gradi = list(degree_dict.values())
    plt.figure()
    plt.hist(gradi, bins=range(min(gradi), max(gradi) + 2), align='left', edgecolor='black')
    plt.title("Distribuzione dei gradi")
    plt.xlabel("Grado")
    plt.ylabel("Numero di nodi")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(nome_file, dpi=150)
    plt.close()


def salva_scatter_grado_betweenness(degree_dict, betweenness, nome_file="./statistiche/grado_vs_betweenness.png"):
    nodi = list(degree_dict.keys())
    x = [degree_dict[n] for n in nodi]
    y = [betweenness[n] for n in nodi]
    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    plt.title("Grado vs. Betweenness Centrality")
    plt.xlabel("Grado")
    plt.ylabel("Betweenness Centrality")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(nome_file, dpi=150)
    plt.close()


def salva_istogramma_clustering(clustering, nome_file="./statistiche/istogramma_clustering.png"):
    valori = list(clustering.values())
    plt.figure()
    plt.hist(valori, bins=10, edgecolor='black')
    plt.title("Distribuzione del Coefficiente di Clustering")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Numero di nodi")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(nome_file, dpi=150)
    plt.close()


def salva_scatter_grado_clustering(degree_dict, clustering, nome_file="./statistiche/grado_vs_clustering.png"):
    nodi = list(degree_dict.keys())
    x = [degree_dict[n] for n in nodi]
    y = [clustering[n] for n in nodi]

    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    plt.title("Grado vs. Clustering Coefficient")
    plt.xlabel("Grado")
    plt.ylabel("Clustering Coefficient")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(nome_file, dpi=150)
    plt.close()


def salva_visualizzazione_grafo(G, degree_dict, clustering, nome_file="./statistiche/rete_clustering.png"):  # noqa
    pos = nx.spring_layout(G, seed=42)
    size_list = [degree_dict[n] * 300 for n in G.nodes()]
    cmap_values = [clustering[n] for n in G.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=size_list,
        node_color=cmap_values,
        alpha=0.8,
        cmap=plt.cm.viridis  # noqa
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Rete: size ~ Degree, colore ~ Clustering coefficient")
    plt.colorbar(nodes, label="Clustering coefficient")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(nome_file, dpi=150)
    plt.close()


def salva_boxplot_metriche(degree_dict, deg_centrality, betweenness, clustering,
                           nome_file="./statistiche/boxplot_metriche.png"):
    dati = [
        list(degree_dict.values()),
        list(deg_centrality.values()),
        list(betweenness.values()),
        list(clustering.values())
    ]
    labels = ["Grado", "Degree Centrality", "Betweenness", "Clustering"]

    plt.figure()
    plt.boxplot(dati, labels=labels, patch_artist=True, showfliers=False)
    plt.title("Boxplot delle principali metriche di grafo")
    plt.ylabel("Valore metrica")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(nome_file, dpi=150)
    plt.close()


def salva_boxplot_individuale(dati, label, nome_file):
    plt.figure()
    plt.boxplot(dati, vert=True, patch_artist=True, showfliers=False)
    plt.title(f"Boxplot: {label}")
    plt.ylabel(label)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(nome_file, dpi=150)
    plt.close()


def main():
    G = nx.read_edgelist("data/congress.edgelist", nodetype=int)

    degree_dict, deg_centrality, betw, clust = calcola_statistiche(G)

    salva_istatogramma_gradi(degree_dict)
    salva_scatter_grado_betweenness(degree_dict, betw)
    salva_istogramma_clustering(clust)
    salva_scatter_grado_clustering(degree_dict, clust)
    salva_visualizzazione_grafo(G, degree_dict, clust)

    salva_boxplot_metriche(degree_dict, deg_centrality, betw, clust)

    salva_boxplot_individuale(list(degree_dict.values()), "Grado", "./statistiche/boxplot_grado.png")
    salva_boxplot_individuale(list(deg_centrality.values()), "Degree Centrality",
                              "./statistiche/boxplot_deg_centrality.png")
    salva_boxplot_individuale(list(betw.values()), "Betweenness", "./statistiche/boxplot_betweenness.png")
    salva_boxplot_individuale(list(clust.values()), "Clustering", "./statistiche/boxplot_clustering.png")

    # nx.write_graphml(G, "rete_con_attributi.graphml")

    print("Analisi completata e file salvati nella cartella statistiche.")


if __name__ == "__main__":
    main()
