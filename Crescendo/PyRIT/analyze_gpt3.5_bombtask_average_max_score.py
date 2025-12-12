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
        "GPT-3.5 + misrepresentation": "experiment_results/batch_2025-12-01_15-49-14",
        "GPT-3.5 + logical appeal": "experiment_results/batch_logical_appeal_gpt-3.5-turbo_2025-12-09_18-40-31",
        "GPT-3.5 + authority endorsement": "experiment_results/batch_authority_endorsement_gpt-3.5-turbo_2025-12-09_03-27-24",
        "GPT-3.5 + expert endorsement": "experiment_results/batch_expert_endorsement_gpt-3.5-turbo_2025-12-09_15-19-38",
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
    models = result_df["Model"].tolist()
    harm_vals = result_df["Harm"].tolist()
    violence_vals = result_df["Violence"].tolist()

    n = len(models)
    gap = 1  # spacing between Harm group and Violence group

    harm_x = np.arange(n)
    viol_x = np.arange(n) + n + gap

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(harm_x, harm_vals, width=0.8, label="Harm")
    bars2 = ax.bar(viol_x, violence_vals, width=0.8, label="Violence")

    # Add text labels
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # X ticks: show model names under each group
    xticks = np.concatenate([harm_x, viol_x])
    xlabels = models + models
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=10)

    # Separator + group labels
    ax.axvline(n - 0.5 + gap / 2, linestyle="--")
    ax.text(harm_x.mean(), -0.12, "", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11)
    ax.text(viol_x.mean(), -0.12, "", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11)

    ax.set_ylabel("Average Max Per-Turn Score")
    ax.set_title("Bomb Task — Max Per-Turn")
    ax.legend()
    plt.tight_layout()

    # Save
    out_dir = "bomb_summary_compare"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "gpt3.5_bomb_max_score_per_turn.png")
    plt.savefig(out_path, dpi=300)

    plt.show()

    print("\nPlot saved as bomb_max_score_per_turn.png")


if __name__ == "__main__":
    main()
