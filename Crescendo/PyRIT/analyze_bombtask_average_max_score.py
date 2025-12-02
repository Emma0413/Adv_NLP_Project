import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_max_from_turns(value):
    """Extract max float from a '0.25|0.00|0.75' style string."""
    if not isinstance(value, str) or value.strip() == "":
        return 0.0
    try:
        parts = [float(x) for x in value.split("|") if x.strip() != ""]
        return max(parts) if parts else 0.0
    except:
        return 0.0


def main():
    # -----------------------------------------------
    # 1. Configure model folders
    # -----------------------------------------------
    batches = {
        "GPT-3.5": "experiment_results/batch_2025-12-01_04-38-37",
        "GPT-4o": "experiment_results/batch_2025-12-01_06-11-45",
        "GPT-3.5 + misrepresentation": "experiment_results/batch_2025-12-01_15-49-14",
        "GPT-4o + misrepresentation": "experiment_results/batch_2025-12-01_08-14-08",
    }

    results = []

    # -----------------------------------------------
    # 2. Load each model’s Bomb task rows
    # -----------------------------------------------
    for label, folder in batches.items():
        csv_path = os.path.join(folder, "summary.csv")
        df = pd.read_csv(csv_path)

        bomb = df[df["task"] == "Bomb"].copy()

        bomb["max_harm"] = bomb["harm_per_turn"].apply(extract_max_from_turns)
        bomb["max_violence"] = bomb["violence_per_turn"].apply(extract_max_from_turns)

        avg_harm = bomb["max_harm"].mean()
        avg_violence = bomb["max_violence"].mean()

        results.append([label, avg_harm, avg_violence])

    # Convert to DataFrame
    result_df = pd.DataFrame(results, columns=["Model", "Harm", "Violence"])
    print(result_df)

    # -----------------------------------------------
    # 3. Plot: X-axis = Harm, Violence
    # -----------------------------------------------
    categories = ["Harm", "Violence"]
    models = result_df["Model"].tolist()

    harm_vals = result_df["Harm"].tolist()
    violence_vals = result_df["Violence"].tolist()

    values = np.array([harm_vals, violence_vals])  # shape: (2 categories × n_models)

    n_categories = len(categories)
    n_models = len(models)

    x = np.arange(n_categories)
    width = 0.12  # width per bar (adjust for more models)

    plt.figure(figsize=(12, 6))

    # Plot each model
    for i in range(n_models):
        plt.bar(
            x + (i - n_models/2) * width + width/2,
            values[:, i],
            width=width,
            label=models[i]
        )

        # Add text labels
        for j, val in enumerate(values[:, i]):
            plt.text(
                x[j] + (i - n_models/2) * width + width/2,
                val,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.xticks(x, categories, fontsize=12)
    plt.ylabel("Average Max Per-Turn Score")
    plt.title("Bomb Task — Harm & Violence (Max Per-Turn)")
    plt.legend(title="")
    plt.tight_layout()

    # Save
    out_dir = "bomb_summary_compare"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "bomb_max_score_per_turn.png")
    plt.savefig(out_path, dpi=300)

    plt.show()

    print("\nPlot saved as bomb_max_score_per_turn.png")


if __name__ == "__main__":
    main()
