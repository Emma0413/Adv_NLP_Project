import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict

def main():
    # --------------------------------------------------------
    # 1. Configure all the batches you want to compare here
    # --------------------------------------------------------
    batches: Dict[str, str] = {
        "GPT 3.5": os.path.join("experiment_results", "batch_2025-12-01_04-38-37"),
        "GPT 4o": os.path.join("experiment_results", "batch_2025-12-01_06-11-45"),
        "GPT 3.5 + misrepresentation": os.path.join("experiment_results", "batch_2025-12-01_15-49-14"),
        "GPT 4o + misrepresentation": os.path.join("experiment_results", "batch_2025-12-01_08-14-08"),
    }

    # --------------------------------------------------------
    # 2. Load CSVs and compute:
    #    - avg actual_turns (SUCCESS only)
    #    - success %
    # --------------------------------------------------------
    avg_list = []
    success_list = []

    for label, folder in batches.items():
        csv_file = os.path.join(folder, "summary.csv")
        df = pd.read_csv(csv_file)

        # SUCCESS mask
        mask_success = df["outcome"] == "AttackOutcome.SUCCESS"
        df_success = df[mask_success]

        # Average turns (SUCCESS only)
        avg = df_success.groupby("task")["actual_turns"].mean()
        avg.name = label
        avg_list.append(avg)

        # Success rate (%)
        success_count = df_success.groupby("task")["outcome"].count()
        total_count = df.groupby("task")["outcome"].count()
        success_rate = (success_count / total_count * 100).rename(label)
        success_list.append(success_rate)

    # Combine the averages and success rates
    all_avgs = pd.concat(avg_list, axis=1)
    all_success = pd.concat(success_list, axis=1)

    all_avgs = all_avgs.sort_index()
    all_success = all_success.reindex(all_avgs.index)

    tasks = all_avgs.index.tolist()
    batch_labels = all_avgs.columns.tolist()

    # --------------------------------------------------------
    # 3. Plot grouped bar chart — ONLY success % shown
    # --------------------------------------------------------
    import numpy as np

    n_tasks = len(tasks)
    n_batches = len(batch_labels)
    x = np.arange(n_tasks)

    total_bar_width = 0.9
    bar_width = total_bar_width / n_batches

    plt.figure(figsize=(14, 6))

    for i, label in enumerate(batch_labels):
        offsets = x - total_bar_width / 2 + bar_width * (i + 0.5)

        success_vals = all_success[label].values

        bars = plt.bar(offsets, all_avgs[label].values, width=bar_width, label=label)

        # Add ONLY success % above bars
        for j, bar in enumerate(bars):
            success = success_vals[j]
            if pd.isna(success):
                text = "N/A"
            else:
                text = f"{success:.0f}%"

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black"
            )

    plt.xticks(x, tasks, rotation=45, ha="right")
    plt.xlabel("Task")
    plt.ylabel("Average Actual Turns (SUCCESS only)")
    plt.title("Success Rate per Task")
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # 4. Save figure
    # --------------------------------------------------------
    output_folder = "summary_comparisons"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "success_rate_compare.png")
    plt.savefig(output_path, dpi=300)

    print(f"Figure saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
