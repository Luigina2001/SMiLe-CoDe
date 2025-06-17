import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Directory output per le immagini
PLOTS_DIR = os.path.join(project_root, "final_graphs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- CONFIGURAZIONE FILE E HEADERS ---
RESULTS_HEADERS = [
    "timestamp", "algorithm_name", "seed_set", "seed_size",
    "final_influence", "final_influence_size",
    "num_nodes", "num_edges", "experiment_result_row",
    "execution_time", "round", "additional_info"
]
COST_HEADERS = [
    "timestamp", "algorithm_name", "cost_function", "use_threshold",
    "budget", "num_nodes", "num_edges", "seed_set",
    "num_seeds", "total_cost", "execution_time", "additional_info"
]

RESULTS_TEMPLATE = "../algorithms/logs/cascade_results/cost{cost}_{algo}_results.csv"
COST_TEMPLATE = "../algorithms/logs/cost{cost}_{algo}.csv"

CHUNKSIZE = 50000  # per la lettura in chunk


def load_results(algorithm=None, cost=None):
    # Determinazione delle coppie (algoritmo, costo) da caricare
    pairs = []
    if algorithm and cost:
        pairs = [(algorithm, cost)]
    elif algorithm:
        pairs = [(algorithm, c) for c in [1, 2, 3]]
    elif cost:
        algos = ["CSG", "WTSS", "SMiLe-CoDe"]
        pairs = [(a, cost) for a in algos]
    else:
        raise ValueError("Devi specificare almeno --algorithm o --cost")

    dfs = []
    for algo, c in pairs:
        path = RESULTS_TEMPLATE.format(cost=c, algo=algo)
        if os.path.exists(path):
            df = pd.read_csv(
                path,
                names=RESULTS_HEADERS,
                header=0,
                usecols=["experiment_result_row", "final_influence_size"]
            )
            df['algorithm'] = algo
            df['cost'] = c
            dfs.append(df)
        else:
            print(f"[WARN] Risultati non trovato: {path}", file=sys.stderr)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_costs(algorithm=None, cost=None):
    # Determinazione delle coppie (algoritmo, costo) da caricare
    pairs = []
    if algorithm and cost:
        pairs = [(algorithm, cost)]
    elif algorithm:
        pairs = [(algorithm, c) for c in [1, 2, 3]]
    elif cost:
        algos = ["CSG", "WTSS", "SMiLe-CoDe"]
        pairs = [(a, cost) for a in algos]
    else:
        raise ValueError("Devi specificare almeno --algorithm o --cost")

    dfs = []
    for algo, c in pairs:
        path = COST_TEMPLATE.format(cost=c, algo=algo)
        if os.path.exists(path):
            dfc = pd.read_csv(
                path,
                names=COST_HEADERS,
                header=0,
                usecols=["budget"]
            )
            dfc = dfc.reset_index().rename(columns={"index": "experiment_result_row"})
            dfc['algorithm'] = algo
            dfc['cost'] = c
            dfs.append(dfc[["experiment_result_row", "budget", "algorithm", "cost"]])
        else:
            print(f"[WARN] Costs non trovato: {path}", file=sys.stderr)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def plot_budget_vs_influence(df, title, save_path):
    """
    Crea un plot con una o più serie in base ai parametri:
    - Solo costo: 1 linea per ogni algoritmo
    - Solo algoritmo: 1 linea per ogni costo
    - Entrambi: 1 linea singola
    """
    plt.figure(figsize=(10, 6))

    # CASO 1: Solo costo specificato (confronto tra algoritmi)
    if 'algorithm' in df.columns and df['algorithm'].nunique() > 1:
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            agg = algo_df.groupby("budget")["final_influence_size"].mean().reset_index()
            plt.plot(agg["budget"], agg["final_influence_size"], marker="o", label=algo)
        plt.legend(title="Algorithm")

    # CASO 2: Solo algoritmo specificato (confronto tra costi)
    elif 'cost' in df.columns and df['cost'].nunique() > 1:
        for cost_val in df['cost'].unique():
            cost_df = df[df['cost'] == cost_val]
            agg = cost_df.groupby("budget")["final_influence_size"].mean().reset_index()
            plt.plot(agg["budget"], agg["final_influence_size"], marker="o", label=f"Cost {cost_val}")
        plt.legend(title="Cost Function")

    # CASO 3: Entrambi specificati o dati singoli
    else:
        agg = df.groupby("budget")["final_influence_size"].mean().reset_index()
        plt.plot(agg["budget"], agg["final_influence_size"], marker="o")

    plt.xlabel("Budget")
    plt.ylabel("Media Final Influence Size")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot salvato in: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grafico: final_influence_size in funzione del budget."
    )
    parser.add_argument("--algorithm", type=str,
                        choices=["CSG", "WTSS", "SMiLe-CoDe"],
                        help="Algoritmo da caricare")
    parser.add_argument("--cost", type=int, choices=[1, 2, 3],
                        help="ID della funzione di costo (1,2,3)")
    args = parser.parse_args()

    # Validazione input
    if not args.algorithm and not args.cost:
        parser.error("Specificare almeno un algoritmo o una funzione di costo")

    try:
        df_res = load_results(algorithm=args.algorithm, cost=args.cost)
        df_costs = load_costs(algorithm=args.algorithm, cost=args.cost)

        if df_res.empty or df_costs.empty:
            raise ValueError("Nessun dato disponibile per i parametri richiesti")

        # Unione dei dataframe
        df = pd.merge(
            df_costs,
            df_res,
            on=["experiment_result_row", "algorithm", "cost"],
            how="inner"
        ).drop_duplicates()

        # DEBUG: campiona fino a 50 righe se richiesto
        if os.getenv("DEBUG") == "1":
            max_sample = min(50, len(df))
            if max_sample > 0:
                print(f"[DEBUG] Campionamento di {max_sample} righe (su {len(df)} totali)")
                df = df.sample(n=max_sample, random_state=42)

        # Costruzione titolo e filename
        title = "Final Influence Size vs Budget"
        fname = "influence_vs_budget"

        if args.algorithm:
            title += f" - {args.algorithm}"
            fname += f"_{args.algorithm}"
        if args.cost:
            title += f" - Cost {args.cost}"
            fname += f"_cost{args.cost}"

        fname += ".png"
        save_path = os.path.join(PLOTS_DIR, fname)

        plot_budget_vs_influence(df, title, save_path)

    except Exception as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()