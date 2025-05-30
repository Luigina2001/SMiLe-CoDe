import networkx as nx
from tqdm import tqdm
import random


# Funzione calcolo della ceiling
def ceildiv(a, b):
    return -(a // -b)


def WTSS(G: nx.Graph, t: dict, c: dict, budget: int):  # noqa
    """
    Input:
      - G: grafo non orientato (nx.Graph)
      - t: dict di thresholds, t[v] = soglia di v
      - c: dict di costs, c[v] = costo di v
      - budget: costo massimo totale ammissibile

    Output:
      - S: target set con costo totale <= budget
    """

    V = set(G.nodes())
    U = set(G.nodes())
    S = set()  # noqa
    total_cost = 0

    # delta[v] = grado corrente di v in U (inizialmente grado in G)
    delta = {v: G.degree(v) for v in V}
    # k[v] = t[v] (threshold residua)
    k = dict(t)
    # N[v] = neighbors di v ancora in U
    N = {v: set(G.neighbors(v)) for v in V}

    pbar = tqdm(total=len(V), desc="WTSS progress")

    def remove_vertex(v):
        """Rimuove v da U, aggiorna delta e N dei suoi neighbors."""
        for u in list(N[v]):  # noqa
            # aggiorna grado residuo e lista di vicini
            delta[u] -= 1
            N[u].remove(v)
        U.remove(v)
        N[v].clear()
        pbar.update(1)

    while U:  # While U ≠ 0 do
        zero_thr = [v for v in U if k[v] == 0]
        if zero_thr:  # if there exists v ∈ U s.t. k(v)=0 then
            v = zero_thr[0]
            # Case 1: the selected vertex v is activated by the influence of its neighbors in V − U only;
            # it can then influence its neighbors in U
            for u in N[v]:
                k[u] = max(0, k[u] - 1)  # noqa
            remove_vertex(v)
            continue

        impossible = [v for v in U if delta[v] < k[v]]
        if impossible:  # if there exists v ∈ U s.t. δ(v) < k(v) then
            # Case 2: the vertex v is added to S, since no sufficient neighbors remain in U to activate it;
            # v can then influence its neighbors in U
            v = impossible[0]
            if total_cost + c[v] <= budget:
                S.add(v)
                total_cost += c[v]
                print(f"Total cost = {total_cost}")
            for u in N[v]:
                k[u] = max(0, k[u] - 1)  # riduco la threshold dei vicini
            remove_vertex(v)
            continue

        # Case 3: the selected vertex v will be activated by its neighbors in U
        def score(v):
            return c[v] * k[v] / (delta[v] * (delta[v] + 1))

        v = max(U, key=score)
        remove_vertex(v)

    pbar.close()
    return S


if __name__ == "__main__":
    G = nx.read_edgelist("data/rete_sociale.txt", nodetype=int)

    budget_k = 10

    # funzione di costo = grado del nodo / 2
    cost1 = {v: ceildiv(G.degree(v), 2) for v in G.nodes()}

    # todo: range da modificare
    # funzione di costo randomica
    cost2 = {v: random.randint(1, max(cost1.values())) for v in G.nodes()}

    cost3 = {v: 1/G.degree(v) for v in G.nodes()}

    # funzione di soglia = floor(grado/2)
    threshold = {v: G.degree(v) // 2 for v in G.nodes()}

    nx.set_node_attributes(G, cost2, "cost")
    nx.set_node_attributes(G, threshold, "threshold")

    S = WTSS(G, threshold, cost2, budget_k)
    print("Target set S =", S)
    print("Total cost =", sum(cost2[v] for v in S))
