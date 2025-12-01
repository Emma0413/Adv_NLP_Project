import asyncio
import logging
import datetime
import csv
import os
from dotenv import load_dotenv

from pyrit.memory.sqlite_memory import SQLiteMemory
from pyrit.memory.central_memory import CentralMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.executor.attack.core import AttackAdversarialConfig, AttackScoringConfig
from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack 
from pyrit.score.true_false.self_ask_true_false_scorer import SelfAskTrueFalseScorer, TrueFalseQuestion
from pyrit.score.float_scale.self_ask_likert_scorer import (
    SelfAskLikertScorer,
    LikertScalePaths,
)
from pyrit.score.batch_scorer import BatchScorer
from task_configs import TASK_CONFIGS



load_dotenv()

# --- Parameters ---
MAX_TURNS_LIST = [3, 5]
MAX_BACKTRACKS_LIST = [0, 5, 10]    

TASK_NAME = "GovHack"                     # <-- which task we’re running
TASK_CFG = TASK_CONFIGS[TASK_NAME]

OBJECTIVE = TASK_CFG["objective"]         # <- taken from task_configs.py



# Metric names for this task (e.g. ["cyber", "exploits"] for GovHack)
LIKERT_METRIC_NAMES = [m["name"] for m in TASK_CFG.get("likert_metrics", [])]

# --- Helper: Universal Text Extractor ---
def get_message_content(entry):
    """Safe text extractor for logs and CSV."""
    possible_attrs = ["content", "value", "original_value", "text", "converted_value"]
    for attr in possible_attrs:
        if hasattr(entry, attr):
            val = getattr(entry, attr)
            if val: return str(val)
    return str(entry)

# --- Helper: Save Clean Log ---
def save_readable_log(conversation_id, file_path, result):
    memory = CentralMemory.get_memory_instance()
    entries = memory.get_conversation(conversation_id=conversation_id)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Experiment Transcript ===\n")
        f.write(f"Objective: {OBJECTIVE}\n")
        f.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")

        for entry in entries:
            if hasattr(entry, "role") and str(entry.role).lower() == "system":
                continue
            
            role_display = "ATTACKER" if hasattr(entry, "role") and str(entry.role).lower() == "user" else "TARGET"
            f.write(f"[{role_display}]\n")
            f.write(f"{get_message_content(entry)}\n")
            f.write("-" * 30 + "\n")

        f.write("\n" + "="*50 + "\n")
        f.write("=== FINAL RESULT ===\n")
        f.write(f"Outcome: {result.outcome}\n")
        f.write(f"Turns Used: {result.executed_turns}\n")
        f.write(f"Backtracks: {result.backtrack_count}\n")
        f.write("="*50 + "\n")

def build_scorers_for_task(
    task_cfg: dict,
    judge_llm: OpenAIChatTarget,
    likert_scorer_cls=SelfAskLikertScorer,
):
    # 1) Objective scorer (must be TrueFalseScorer for AttackScoringConfig)
    tf_question: TrueFalseQuestion = task_cfg["tf_question"]
    objective_scorer = SelfAskTrueFalseScorer(
        chat_target=judge_llm,
        true_false_question=tf_question,
    )

    # 2) Likert scorers (can be many, all using the same class by default)
    likert_metrics = task_cfg.get("likert_metrics", [])
    likert_scorers = {
        m["name"]: likert_scorer_cls(
            chat_target=judge_llm,
            likert_scale_path=m["scale"].value,
        )
        for m in likert_metrics
    }

    scoring_config = AttackScoringConfig(
        objective_scorer=objective_scorer,
        # Register Likert scorers as auxiliary_scorers (optional but nice for PyRIT’s internals)
        auxiliary_scorers=list(likert_scorers.values()),
    )

    return scoring_config, objective_scorer, likert_scorers


async def run_single_experiment(max_turns, max_backtracks, experiment_id, output_folder):
    log_filename = f"exp_{experiment_id:02d}_Turns{max_turns}_Backtracks{max_backtracks}.txt"
    full_log_path = os.path.join(output_folder, log_filename)

    print(
        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
        f"Running Exp {experiment_id} -> {log_filename}"
    )

    CentralMemory.set_memory_instance(SQLiteMemory())

    # Roles
    victim = OpenAIChatTarget(model_name="gpt-3.5-turbo")
    adversary = OpenAIChatTarget(model_name="gpt-4o-mini")
    judge_llm = OpenAIChatTarget(model_name="gpt-4o-mini")

    # --- OBJECTIVE SCORER (True/False, needed by AttackScoringConfig) ---

    scoring_config, tf_scorer, likert_scorers = build_scorers_for_task(
        TASK_CFG,
        judge_llm,
    )


    crescendo = CrescendoAttack(
        objective_target=victim,
        attack_adversarial_config=AttackAdversarialConfig(target=adversary),
        attack_scoring_config=scoring_config,
        max_turns=max_turns,
        max_backtracks=max_backtracks,
    )
    try:
        result = await crescendo.execute_async(objective=OBJECTIVE)

        # --- Per-turn Likert float scores (post-hoc analysis for ALL metrics) ---
        memory = CentralMemory.get_memory_instance()
        all_pieces = memory.get_message_pieces(conversation_id=result.conversation_id)

        # Only score assistant outputs (i.e., victim model responses)
        response_pieces = [
            p
            for p in all_pieces
            if hasattr(p, "role") and str(p.role).lower() == "assistant"
        ]

        per_turn_scores = {name: [] for name in likert_scorers.keys()}
        last_scores = {name: None for name in likert_scorers.keys()}

        if response_pieces and likert_scorers:
            prompt_ids = [str(p.id) for p in response_pieces]
            batch_scorer = BatchScorer()

            for metric_name, scorer in likert_scorers.items():
                scores = await batch_scorer.score_responses_by_filters_async(
                    scorer=scorer,
                    prompt_ids=prompt_ids,
                )

                score_map = {str(s.message_piece_id): s for s in scores}
                ordered_scores = [score_map[pid] for pid in prompt_ids if pid in score_map]

                # Use 0–1 float values (score_value). If you want raw 1–5, use s.score_metadata["likert_value"].
                vals = [float(s.score_value) for s in ordered_scores]
                per_turn_scores[metric_name] = vals
                if vals:
                    last_scores[metric_name] = vals[-1]

        # Save the human-readable log
        save_readable_log(result.conversation_id, full_log_path, result)

        preview_text = "No response"
        if result.last_response:
            preview_text = get_message_content(result.last_response)

        # Base row
        row = {
            "id": experiment_id,
            "task": TASK_NAME,
            "max_turns": max_turns,
            "max_backtracks": max_backtracks,
            "outcome": str(result.outcome),
            "actual_turns": result.executed_turns,
            "actual_backtracks": result.backtrack_count,
            "response_preview": preview_text[:100].replace("\n", " "),
        }

        # Add one pair of columns per metric: <name>_per_turn, <name>_last
        for metric_name in LIKERT_METRIC_NAMES:
            vals = per_turn_scores.get(metric_name, [])
            last_val = last_scores.get(metric_name, None)

            per_turn_str = "|".join(f"{v:.3f}" for v in vals) if vals else ""
            last_str = f"{last_val:.3f}" if last_val is not None else ""

            row[f"{metric_name}_per_turn"] = per_turn_str
            row[f"{metric_name}_last"] = last_str

        return row

    except Exception as e:
        with open(full_log_path, "w", encoding="utf-8") as f:
            f.write(f"!!! CRASH REPORT !!!\nError: {str(e)}")
        raise e


async def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_folder_name = f"batch_{timestamp}"
    output_folder = os.path.join("experiment_results", batch_folder_name)

    os.makedirs(output_folder, exist_ok=True)
    CentralMemory.set_memory_instance(SQLiteMemory())
    logging.getLogger().setLevel(logging.WARNING)

    print("=== Batch Starting ===")
    print(f"Task: {TASK_NAME}")
    print(f"Results: {output_folder}")

    results = []
    exp_id = 1

    for turns in MAX_TURNS_LIST:
        for backtracks in MAX_BACKTRACKS_LIST:
            try:
                data = await run_single_experiment(
                    turns, backtracks, exp_id, output_folder
                )
                results.append(data)
            except Exception as e:
                print(f"!!! Error in Exp {exp_id}: {e}")
                # Build error row with empty metrics
                row = {
                    "id": exp_id,
                    "task": TASK_NAME,
                    "max_turns": turns,
                    "max_backtracks": backtracks,
                    "outcome": "ERROR",
                    "actual_turns": 0,
                    "actual_backtracks": 0,
                    "response_preview": f"Error: {str(e)}",
                }
                for metric_name in LIKERT_METRIC_NAMES:
                    row[f"{metric_name}_per_turn"] = ""
                    row[f"{metric_name}_last"] = ""
                results.append(row)
            await asyncio.sleep(5)
            exp_id += 1

    # Build CSV header dynamically based on metrics
    base_fields = [
        "id",
        "task",
        "max_turns",
        "max_backtracks",
        "outcome",
        "actual_turns",
        "actual_backtracks",
        "response_preview",
    ]
    metric_fields = []
    for metric_name in LIKERT_METRIC_NAMES:
        metric_fields.append(f"{metric_name}_per_turn")
        metric_fields.append(f"{metric_name}_last")

    fieldnames = base_fields + metric_fields

    csv_path = os.path.join(output_folder, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n=== All Done ===")
    print(f"Folder: {output_folder}")


if __name__ == "__main__":
    asyncio.run(main())