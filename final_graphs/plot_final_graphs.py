import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import seaborn as sns

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
    if algorithm and cost:
        pairs = [(algorithm, cost)]
    elif algorithm:
        pairs = [(algorithm, c) for c in [1, 2, 3]]
    elif cost:
        algos = ["CSG", "WTSS", "SMiLe-CoDe", "SMiLe-CoDe-bridges"]
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
                usecols=["experiment_result_row", "final_influence_size", "execution_time"]
            )
            df['algorithm'] = algo
            df['cost'] = c
            dfs.append(df)
        else:
            print(f"[WARN] Risultati non trovato: {path}", file=sys.stderr)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_costs(algorithm=None, cost=None):
    # Determinazione delle coppie (algoritmo, costo) da caricare
    if algorithm and cost:
        pairs = [(algorithm, cost)]
    elif algorithm:
        pairs = [(algorithm, c) for c in [1, 2, 3]]
    elif cost:
        algos = ["CSG", "WTSS", "SMiLe-CoDe", "SMiLe-CoDe-bridges"]
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
                usecols=["budget", "num_seeds", "cost_function"]
            )
            dfc = dfc.reset_index().rename(columns={"index": "experiment_result_row"})
            dfc['algorithm'] = algo
            dfc['cost'] = c
            dfs.append(dfc[["experiment_result_row", "budget", "num_seeds", "cost_function",
                            "algorithm", "cost"]])
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
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8')
    all_budgets = sorted(df["budget"].unique())

    def fill_missing(agg_df):
        """Riempie i valori mancanti estendendo l'ultimo punto disponibile."""
        agg_df = agg_df.set_index("budget").reindex(all_budgets).ffill().reset_index()
        return agg_df

    # Configurazione degli stili per le linee
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    colors = [
        '#ff7f0e',  # arancione
        '#17becf',  # ciano
        '#1f77b4',  # blu
        '#98df8a',  # verde chiaro 2
        '#ffbb78',  # arancione chiaro
        '#2ca02c',  # verde
        '#d62728',  # rosso
        '#8c564b',  # marrone
        '#e377c2',  # rosa
        '#7f7f7f',  # grigio
        '#bcbd22',  # verde chiaro 1
        '#aec7e8',  # blu chiaro
    ]

    # Funzione per determinare dove posizionare i marker
    def get_marker_positions(length, step=3):
        return [k for k in range(length) if k % step == 0 or k == length - 1]

    # CASO 1: Confronto tra algoritmi
    if 'algorithm' in df.columns and df['algorithm'].nunique() > 1:
        for i, algo in enumerate(df['algorithm'].unique()):
            algo_df = df[df['algorithm'] == algo]
            agg = algo_df.groupby("budget")["final_influence_size"].mean().reset_index()
            agg = fill_missing(agg)

            marker_pos = get_marker_positions(len(agg))

            plt.plot(agg["budget"], agg["final_influence_size"],
                     linestyle=line_styles[i % len(line_styles)],
                     color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     markevery=marker_pos,
                     markersize=8,
                     linewidth=2.5,
                     label=algo)

        plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"{title}\nConfronto tra Algoritmi", pad=20)

    # CASO 2: Confronto tra costi
    elif 'cost' in df.columns and df['cost'].nunique() > 1:
        for i, cost_val in enumerate(sorted(df['cost'].unique())):
            cost_df = df[df['cost'] == cost_val]
            agg = cost_df.groupby("budget")["final_influence_size"].mean().reset_index()
            agg = fill_missing(agg)

            marker_pos = get_marker_positions(len(agg))

            plt.plot(agg["budget"], agg["final_influence_size"],
                     linestyle=line_styles[i % len(line_styles)],
                     color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     markevery=marker_pos,
                     markersize=8,
                     linewidth=2.5,
                     label=f"Cost {cost_val}")

        plt.legend(title="Cost Function", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"{title}\nConfronto tra Funzioni di Costo", pad=20)

    # CASO 3: Singola serie
    else:
        agg = df.groupby("budget")["final_influence_size"].mean().reset_index()
        agg = fill_missing(agg)

        marker_pos = get_marker_positions(len(agg))

        plt.plot(agg["budget"], agg["final_influence_size"],
                 linestyle='-',
                 color=colors[0],
                 marker='o',
                 markevery=marker_pos,
                 markersize=8,
                 linewidth=2.5)

        plt.title(title, pad=20)

    plt.xlabel("Budget", fontsize=12, labelpad=10)
    plt.ylabel("Influenza Media Finale", fontsize=12, labelpad=10)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if 'algorithm' in df.columns or 'cost' in df.columns:
        plt.subplots_adjust(right=0.8)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico salvato in: {save_path}")


def plot_execution_time_comparison(df, title, save_path):
    """
    Confronta i tempi di esecuzione degli algoritmi
    """
    plt.figure(figsize=(12, 7))

    # Prepara i dati
    if 'algorithm' in df.columns and df['algorithm'].nunique() > 1:
        sns.boxplot(x='algorithm', y='execution_time', data=df)
        plt.xticks(rotation=45)
        plt.title(f"{title}\nConfronto Tempi di Esecuzione", pad=20)
        plt.xlabel("Algoritmo")
        plt.ylabel("Tempo di esecuzione (s)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Grafico salvato in: {save_path}")


def plot_budget_vs_seed_size(df, title, save_path):
    """
    Mostra come varia il numero di seed in base al budget
    """
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8')

    if 'budget' in df.columns and 'num_seeds' in df.columns and 'cost_function' in df.columns:
        colors = [
            '#ff7f0e',  # arancione
            '#17becf',  # ciano
            '#1f77b4',  # blu
            '#98df8a',  # verde chiaro 2
            '#ffbb78',  # arancione chiaro
            '#2ca02c',  # verde
            '#d62728',  # rosso
            '#8c564b',  # marrone
            '#e377c2',  # rosa
            '#7f7f7f',  # grigio
            '#bcbd22',  # verde chiaro 1
            '#aec7e8',  # blu chiaro
        ]

        # Lista di marker distinti per le funzioni di costo
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'd']

        algorithms = df['algorithm'].unique()
        cost_functions = df['cost_function'].unique()

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot per ogni combinazione algoritmo + funzione di costo
        for i, algo in enumerate(algorithms):
            for j, cost_func in enumerate(cost_functions):
                subset = df[(df['algorithm'] == algo) & (df['cost_function'] == cost_func)]
                if not subset.empty:
                    ax.scatter(
                        subset['budget'],
                        subset['num_seeds'],
                        color=colors[i % len(colors)],
                        marker=markers[j % len(markers)],
                        s=120,  # Aumentato leggermente la dimensione
                        edgecolor='black',  # Bordo nero per maggiore contrasto
                        linewidth=0.5,  # Spessore del bordo
                        alpha=0.9,  # Leggera trasparenza
                        label=f"{algo} ({cost_func})" if j == 0 else None
                    )

        from matplotlib.lines import Line2D

        # Legenda per gli algoritmi (colori)
        algo_legend = [
            Line2D([0], [0],
                   marker='o',
                   color='w',
                   markerfacecolor=colors[i % len(colors)],
                   markersize=12,
                   label=algo)
            for i, algo in enumerate(algorithms)
        ]

        # Legenda per le funzioni di costo (marker)
        cost_legend = [
            Line2D([0], [0],
                   marker=markers[i % len(markers)],
                   color='w',
                   markerfacecolor='gray',
                   markersize=12,
                   label=func)
            for i, func in enumerate(cost_functions)
        ]

        first_legend = ax.legend(
            handles=algo_legend,
            title="Algoritmi",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True
        )
        ax.add_artist(first_legend)

        ax.legend(
            handles=cost_legend,
            title="Funzioni di Costo",
            bbox_to_anchor=(1.05, 0.7),
            loc='upper left',
            frameon=True
        )

        # Formattazione del grafico
        plt.title(f"{title}\nRelazione Budget - Dimensione Seed Set", pad=20)
        plt.xlabel("Budget", fontsize=12)
        plt.ylabel("Numero di Seed", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        plt.subplots_adjust(right=0.75)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Grafico salvato in: {save_path}")
    else:
        missing = []
        if 'budget' not in df.columns:
            missing.append('budget')
        if 'num_seeds' not in df.columns:
            missing.append('num_seeds')
        if 'cost_function' not in df.columns:
            missing.append('cost_function')
        print(f"Colonne mancanti nel DataFrame: {', '.join(missing)}", file=sys.stderr)


def plot_influence_distribution(df, title, save_path):
    """
    Mostra la distribuzione dell'influenza finale per algoritmo
    """
    plt.figure(figsize=(12, 7))

    if 'final_influence_size' in df.columns:
        sns.violinplot(x='algorithm', y='final_influence_size', data=df)
        plt.title(f"{title}\nDistribuzione Influenza Finale", pad=20)
        plt.xlabel("Algoritmo")
        plt.ylabel("Dimensione Influenza Finale")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Grafico salvato in: {save_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Grafico: final_influence_size in funzione del budget."
    )
    parser.add_argument("--algorithm", type=str,
                        choices=["CSG", "WTSS", "SMiLe-CoDe", "SMiLe-CoDe-bridges"],
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

        # plot_budget_vs_influence(df, title, save_path)

        base_name = ""
        title = ""
        if args.algorithm:
            title += f"{args.algorithm}"
            base_name += f"{args.algorithm}"
        if args.cost:
            title += f"Cost {args.cost}"
            base_name += f"cost{args.cost}"

        # plot_execution_time_comparison(df, title, os.path.join(PLOTS_DIR, f"{base_name}_execution_time.png"))
        plot_budget_vs_seed_size(df, title, os.path.join(PLOTS_DIR, f"{base_name}_budget_vs_seeds.png"))
        # plot_influence_distribution(df, title, os.path.join(PLOTS_DIR, f"{base_name}_influence_dist.png"))

    except Exception as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
