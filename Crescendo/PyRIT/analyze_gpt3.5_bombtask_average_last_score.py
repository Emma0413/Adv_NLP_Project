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
        "GPT-3.5 + misrepresentation": "experiment_results/batch_2025-12-01_15-49-14",
        "GPT-3.5 + logical appeal": "experiment_results/batch_logical_appeal_gpt-3.5-turbo_2025-12-09_18-40-31",
        "GPT-3.5 + authority endorsement": "experiment_results/batch_authority_endorsement_gpt-3.5-turbo_2025-12-09_03-27-24",
        "GPT-3.5 + expert endorsement": "experiment_results/batch_expert_endorsement_gpt-3.5-turbo_2025-12-09_15-19-38",
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
    batch_labels = list(batches.keys())
    n = len(batch_labels)

    harm_vals = [harm_results.get(b, 0.0) for b in batch_labels]
    viol_vals = [violence_results.get(b, 0.0) for b in batch_labels]

    gap = 1  # space between the two groups
    harm_x = np.arange(n)
    viol_x = np.arange(n) + n + gap

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(harm_x, harm_vals, width=0.8, label="Harm")
    bars2 = ax.bar(viol_x, viol_vals, width=0.8, label="Violence")

    # Value labels
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # X ticks: show batch names twice (once under Harm group, once under Violence group)
    xticks = np.concatenate([harm_x, viol_x])
    xlabels = batch_labels + batch_labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")

    # Visual separator + group labels
    ax.axvline(n - 0.5 + gap / 2, linestyle="--")
    ax.text(harm_x.mean(), -0.12, "", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10)
    ax.text(viol_x.mean(), -0.12, "", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10)

    ax.set_ylabel("Average Score")
    ax.set_title("Bomb Task – Harm & Violence")
    ax.legend()
    plt.tight_layout()

    # Save
    out_dir = "bomb_summary_compare"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "gpt3.5_bomb_last_score.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
