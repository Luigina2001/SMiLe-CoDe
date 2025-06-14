from typing import Callable, Union, Optional, Set, Dict
import heapq
import networkx as nx
from tqdm import tqdm
import time
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import log_experiment, assign_cost_attributes, ceil_division  # noqa


def sub_function1(S: set, G: nx.Graph) -> float:
    """Versione ottimizzata senza loop annidati."""
    if not S:
        return 0
    score = 0
    for v in G.nodes():
        deg = G.degree(v)
        half_deg = ceil_division(deg, 2)
        neighbors_in_S = len(set(G.neighbors(v)) & S)
        score += min(neighbors_in_S, half_deg)
    return score


def sub_function2(S: set, G: nx.Graph) -> float:
    """Ottimizzata con formula matematica."""
    if not S:
        return 0
    score = 0
    for v in G.nodes():
        deg = G.degree(v)
        half_deg = ceil_division(deg, 2)
        neighbors_in_S = len(set(G.neighbors(v)) & S)
        c = min(neighbors_in_S, half_deg)
        if c > 0:
            score += c * half_deg - (c * (c - 1)) // 2
        if neighbors_in_S > half_deg:
            score += (half_deg * (half_deg + 1)) // 2
    return score


def sub_function3(S: set, G: nx.Graph) -> float:
    """Ottimizzata riducendo loop interni."""
    if not S:
        return 0
    score = 0
    for v in G.nodes():
        deg = G.degree(v)
        half_deg = ceil_division(deg, 2)
        neighbors_in_S = len(set(G.neighbors(v)) & S)
        for i in range(1, min(neighbors_in_S, half_deg) + 1):
            denom = deg - i + 1
            if denom > 0:
                score += (half_deg - i + 1) / denom
    return score


def compute_delta(
        sub_func: Callable,
        old_count: int,
        new_count: int,
        half_deg: int,
        deg: Optional[int] = None
) -> float:
    """Calcola la variazione del punteggio per un nodo."""
    if sub_func == sub_function1:
        return min(new_count, half_deg) - min(old_count, half_deg)
    elif sub_func == sub_function2:
        return max(half_deg - old_count, 0) if old_count < half_deg else 0
    elif sub_func == sub_function3:
        if old_count < half_deg and deg is not None and (deg - old_count) > 0:
            return (half_deg - old_count) / (deg - old_count)
        return 0
    return 0


def cost_seeds_greedy(
        G: nx.Graph,
        budget: Union[int, float],
        cost_type: str,
        sub_function: Callable,
        initial_seed_set: Optional[Set] = None,
        current_cost: Union[int, float] = 0
) -> Set[int]:
    """Versione ottimizzata con aggiornamento incrementale e heap."""
    if sub_function not in {sub_function1, sub_function2, sub_function3}:
        raise ValueError("Funzione submodulare non supportata")

    # Precalcolo gradi e half_deg
    degree = dict(G.degree())
    half_deg = {v: ceil_division(deg, 2) for v, deg in degree.items()}

    # Inizializzazione strutture dati
    neighbors_in_S_count: Dict[int, int] = {v: 0 for v in G.nodes}
    current_value = 0
    total_cost = current_cost
    epsilon = 1e-6

    # Costruzione set iniziale
    if initial_seed_set is not None:
        S_selected = set(initial_seed_set)
        for u in initial_seed_set:
            for w in G.neighbors(u):
                old_count = neighbors_in_S_count[w]
                new_count = old_count + 1
                delta = compute_delta(
                    sub_function, old_count, new_count,
                    half_deg[w], degree[w] if sub_function == sub_function3 else None
                )
                current_value += delta
                neighbors_in_S_count[w] = new_count
    else:
        S_selected = set()

    remaining_nodes = set(G.nodes) - S_selected
    best_value = {}
    heap = []

    # Inizializza l'heap
    for v in remaining_nodes:
        gain = 0.0
        for w in G.neighbors(v):
            old_count = neighbors_in_S_count[w]
            new_count = old_count + 1
            delta = compute_delta(
                sub_function, old_count, new_count,
                half_deg[w], degree[w] if sub_function == sub_function3 else None
            )
            gain += delta
        cost_v = G.nodes[v].get(cost_type, 0) or epsilon
        val = gain / cost_v
        best_value[v] = val
        heapq.heappush(heap, (-val, v))

    # Ciclo greedy
    with tqdm(total=budget, initial=total_cost, desc="Cost Seeds Greedy", unit="cost") as pbar:
        while heap and total_cost < budget:
            # Trova il nodo con il miglior rapporto guadagno/costo
            while heap:
                neg_val, v = heapq.heappop(heap)
                if best_value.get(v, float('-inf')) == -neg_val:
                    break
            else:
                break

            cost_v = G.nodes[v].get(cost_type, 0) or epsilon
            if total_cost + cost_v > budget:
                break

            # Aggiungi il nodo al set
            total_cost += cost_v
            pbar.update(cost_v)
            S_selected.add(v)
            remaining_nodes.remove(v)

            # Aggiorna i vicini
            for w in G.neighbors(v):
                old_count = neighbors_in_S_count[w]
                new_count = old_count + 1
                delta = compute_delta(
                    sub_function, old_count, new_count,
                    half_deg[w], degree[w] if sub_function == sub_function3 else None
                )
                current_value += delta
                neighbors_in_S_count[w] = new_count

            # Aggiorna l'heap per i nodi interessati
            affected = set()
            for w in G.neighbors(v):
                for u in G.neighbors(w):
                    if u in remaining_nodes:
                        affected.add(u)

            for u in affected:
                gain = 0.0
                for w in G.neighbors(u):
                    old_count = neighbors_in_S_count[w]
                    new_count = old_count + 1
                    delta = compute_delta(
                        sub_function, old_count, new_count,
                        half_deg[w], degree[w] if sub_function == sub_function3 else None
                    )
                    gain += delta
                cost_u = G.nodes[u].get(cost_type, 0) or epsilon
                new_val = gain / cost_u
                if new_val != best_value.get(u, None):
                    best_value[u] = new_val
                    heapq.heappush(heap, (-new_val, u))

    return S_selected


if __name__ == "__main__":
    G = nx.read_edgelist("../data/facebook_combined.txt", nodetype=int)
    G, cost1, cost2, cost3 = assign_cost_attributes(G, use_threshold=False)

    # Configurazioni funzioni di costo e relative descrizioni
    cost_functions = {
        # "cost1": cost1,
        # "cost2": cost2,
        "cost3": cost3
    }

    descriptions = {
        # "cost1": "cost1: ceiling function of degree(v) / 2",
        # "cost2": "cost2: random int in [min(cost1), max(cost1)]",
        "cost3": "cost3: scaled log10 of betweenness centrality"
    }

    for name, cost in cost_functions.items():
        algorithm_name = "CSG"
        cost_function_desc = descriptions[name]
        desc = descriptions[name]

        # Calcolo range del budget
        min_budget = int(max(cost.values()))
        if int(min(cost.values())) > 0:
            max_budget = int(min(cost.values()) * (len(G.nodes())))
        else:
            max_budget = (int(min(cost.values()) + 1) * (len(G.nodes())))

        if min_budget > max_budget:
            print(f"MinBudget > MaxBudget for {name}")
            min_budget, max_budget = max_budget, min_budget

        if min_budget == max_budget:
            print(f"MinBudget = MaxBudget for {name}")
            continue

        tqdm.write(f"\n{name} — budget da {min_budget} a {max_budget}")

        current_seed_set: Optional[Set[int]] = None
        current_cost = 0

        for budget_k in tqdm(
            range(min_budget, max_budget + 1, 100),
            desc=f"Budget loop for {name}",
            unit="budget"
        ):
            start_time = time.time()
            S = cost_seeds_greedy(G, budget_k, name, sub_function1, current_seed_set, current_cost)
            end_time = time.time()

            total_cost = sum(cost[v] for v in S)
            current_seed_set = S
            current_cost = total_cost

            exec_time = end_time - start_time

            tqdm.write(f"Function: {name} | Budget: {budget_k}")
            tqdm.write(f"Seed set size: {len(S)}; Total cost: {total_cost}; Time: {exec_time:.2f}s")

            log_experiment(
                csv_path=f"./logs/{name}_CSG.csv",
                algorithm_name=algorithm_name,
                cost_function=cost_function_desc,
                use_threshold=False,
                budget=budget_k,
                seed_set=S,
                total_cost=total_cost,
                execution_time=exec_time,
                G=G,
                additional_info={"note": f"Running on facebook_combined.txt with {name}"}
            )
