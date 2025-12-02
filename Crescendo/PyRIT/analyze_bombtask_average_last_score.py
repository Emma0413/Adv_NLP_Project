import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_float_or_zero(x):
    try:
        return float(x)
    except:
        return 0.0

def main():
    # ---------------------------------------
    # 1. Configure batches
    # ---------------------------------------
    batches = {
        "GPT-3.5": "experiment_results/batch_2025-12-01_04-38-37",
        "GPT-4o": "experiment_results/batch_2025-12-01_06-11-45",
        "GPT-3.5 + misrepresentation": "experiment_results/batch_2025-12-01_15-49-14",
        "GPT-4o + misrepresentation": "experiment_results/batch_2025-12-01_08-14-08",
    }

    harm_results = {}
    violence_results = {}

    # ---------------------------------------
    # 2. Load each CSV and compute averages
    # ---------------------------------------
    for label, folder in batches.items():
        csv_path = os.path.join(folder, "summary.csv")

        if not os.path.exists(csv_path):
            print(f"⚠ CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        bomb_df = df[df["task"] == "Bomb"].copy()

        bomb_df["harm_last"] = bomb_df["harm_last"].apply(parse_float_or_zero)
        bomb_df["violence_last"] = bomb_df["violence_last"].apply(parse_float_or_zero)

        harm_results[label] = bomb_df["harm_last"].mean()
        violence_results[label] = bomb_df["violence_last"].mean()

    # ---------------------------------------
    # 3. Build plot (Two x-axis groups: harm, violence)
    # ---------------------------------------
    categories = ["harm_last", "violence_last"]
    x = np.arange(len(categories))

    batch_labels = list(batches.keys())
    n_batches = len(batch_labels)
    width = 0.8 / n_batches  # distribute bars within category

    plt.figure(figsize=(12, 6))

    for i, batch in enumerate(batch_labels):
        offsets = x - 0.4 + i * width + width / 2

        values = [
            harm_results.get(batch, 0.0),
            violence_results.get(batch, 0.0)
        ]

        bars = plt.bar(offsets, values, width=width, label=batch)

        # Add text above bars
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.xticks(x, ["Harm", "Violence"])
    plt.ylabel("Average Score")
    plt.title("Bomb Task – Harm & Violence Comparison")
    plt.legend()
    plt.tight_layout()

    # Save
    out_dir = "bomb_summary_compare"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "bomb_last_score.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
