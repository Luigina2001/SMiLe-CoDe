import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_hist_time_comparison(df, title, save_path):
    plt.figure(figsize=(10, 10))
    bars = plt.bar(df['Algorithm'], df['Execution Time'], color='skyblue', edgecolor='black')

    # Add labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title(title)
    plt.ylabel("Execution Time (s)")
    plt.xlabel("Algorithm")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_execution_times(cost_id):
    # Costruzione dei percorsi in base al costo
    base_path = "../algorithms/logs"
    cost_prefix = f"cost{cost_id}"

    algos = ["CSG", "WTSS", "SMiLe-CoDe", "SMiLe-CoDe-bridges"]
    df_exec = []
    df_casc = []

    for algo in algos:
        try:
            # Seed set execution time
            df_algo = pd.read_csv(os.path.join(base_path, f"{cost_prefix}_{algo}.csv"))
            exec_time = df_algo['execution_time'].sum()
            df_exec.append((algo, exec_time))

            # Cascade execution time
            df_algo_c = pd.read_csv(os.path.join(base_path, "cascade_results", f"{cost_prefix}_{algo}_results.csv"))
            exec_time_c = df_algo_c['execution_time'].sum()
            df_casc.append((algo, exec_time_c))
        except FileNotFoundError as e:
            print(f"[WARN] File non trovato per {algo}: {e}")

    df_alg = pd.DataFrame(df_exec, columns=["Algorithm", "Execution Time"])
    df_c = pd.DataFrame(df_casc, columns=["Algorithm", "Execution Time"])
    return df_alg, df_c


def main():
    parser = argparse.ArgumentParser(description="Confronto dei tempi di esecuzione per funzione di costo")
    parser.add_argument("--cost", type=int, choices=[1, 2, 3], required=True,
                        help="ID della funzione di costo (1, 2, 3)")
    args = parser.parse_args()

    df_alg, df_c = load_execution_times(args.cost)

    plots_dir = "../final_graphs/plots"
    os.makedirs(plots_dir, exist_ok=True)

    plot_hist_time_comparison(
        df_alg,
        f"Total execution time of seed set search (cost {args.cost})",
        os.path.join(plots_dir, f"total_seed_set_time_cost{args.cost}.png")
    )

    plot_hist_time_comparison(
        df_c,
        f"Total execution time of cascade (cost {args.cost})",
        os.path.join(plots_dir, f"total_cascade_time_cost{args.cost}.png")
    )


if __name__ == "__main__":
    main()
