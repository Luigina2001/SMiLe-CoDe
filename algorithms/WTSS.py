import networkx as nx
from tqdm import tqdm
import os
import sys
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import assign_cost_attributes, log_experiment  # noqa


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
            if total_cost + c[v] < budget:
                S.add(v)
                total_cost += c[v]
                print(f"Total cost = {total_cost}")
            elif total_cost + c[v] == budget:  # possibile in quanto non ci sono nodi con costo uguale a zero
                S.add(v)
                total_cost += c[v]
                print(f"Total cost = {total_cost}")
                print(f"BREAK: {total_cost} = {budget}")
                pbar.close()
                return S
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
    G = nx.read_edgelist("../data/facebook_combined.txt", nodetype=int)

    budget_k = 1000
    algorithm_name = "WTSS"
    cost_function_desc = "cost1: ceiling function of degree(v) / 2"

    G, cost1, cost2, cost3, threshold = assign_cost_attributes(G, budget_k, True)

    start_time = time.time()
    S = WTSS(G, threshold, cost1, budget_k)
    end_time = time.time()

    total_cost = sum(cost1[v] for v in S)
    exec_time = end_time - start_time

    print("Target set S =", S)
    print("Total cost =", total_cost)
    print(f"Execution time: {exec_time:.2f} secondi")

    log_experiment(
        csv_path="./logs/experiment_results.csv",
        algorithm_name=algorithm_name,
        cost_function=cost_function_desc,
        use_threshold=True,
        budget=budget_k,
        seed_set=S,
        total_cost=total_cost,
        execution_time=exec_time,
        G=G,
        additional_info={"note": "Esecuzione su facebook_combined.txt"}
    )
