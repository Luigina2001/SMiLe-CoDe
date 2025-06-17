from utils.utils import ceil_division
import networkx as nx

def sub_function1(S: set, G: nx.Graph) -> float:  # noqa
    """
        Input:
          - S: target set di nodi di G
          - G: grafo non orientato (nx.Graph)
        Output:
          - score: punteggio assegnato dalla funzione sub-modulare 1 al set S
    """
    if not S:
        return 0
    score = 0
    for v in G.nodes():
        deg = G.degree(v)
        half_deg = ceil_division(deg, 2)
        neighbors_in_S = len(set(G.neighbors(v)) & S)
        score += min(neighbors_in_S, half_deg)
    return score


def sub_function2(S: set, G: nx.Graph) -> float:  # noqa
    """
        Input:
          - S: target set di nodi di G
          - G: grafo non orientato (nx.Graph)
        Output:
          - score: punteggio assegnato dalla funzione sub-modulare 2 al set S
    """
    if not S:
        return 0

    score = 0
    for v in G.nodes():
        neighbors_in_S = [u for u in G.neighbors(v) if u in S]
        half_deg = ceil_division(G.degree(v), 2)

        for i in range(1, len(neighbors_in_S) + 1):
            score += max(half_deg - i + 1, 0)

    return score

def sub_function3(S: set, G: nx.Graph) -> float:  # noqa
    """
        Input:
          - S: target set di nodi di G
          - G: grafo non orientato (nx.Graph)
        Output:
          - score: punteggio assegnato dalla funzione sub-modulare 3 al set S
    """
    if not S:
        return 0
    score = 0
    for v in G.nodes():
        deg = G.degree(v)
        half_deg = ceil_division(deg, 2)
        neighbors_in_S = len(set(G.neighbors(v)) & S)
        for i in range(1, neighbors_in_S + 1):
            denom = deg - i + 1
            if denom > 0:
                score += max((half_deg - i + 1) / denom, 0)
    return score
