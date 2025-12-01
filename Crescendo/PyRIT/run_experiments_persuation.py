import asyncio
import logging
import datetime
import csv
import os
from dotenv import load_dotenv

from pyrit.memory.sqlite_memory import SQLiteMemory
from pyrit.memory.central_memory import CentralMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.executor.attack.core import (
    AttackAdversarialConfig,
    AttackScoringConfig,
    AttackConverterConfig,
)
from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.prompt_normalizer import PromptConverterConfiguration

from pyrit.score.true_false.self_ask_true_false_scorer import (
    SelfAskTrueFalseScorer,
    TrueFalseQuestion,
)
from pyrit.score.float_scale.self_ask_likert_scorer import SelfAskLikertScorer
from pyrit.score.batch_scorer import BatchScorer
from task_configs import TASK_CONFIGS

# ⬇️ use the built-in persuasion converter
from pyrit.prompt_converter.persuasion_converter import PersuasionConverter

load_dotenv()

# --- Parameters ---
# Use repeated 7s as "replications" per task
MAX_TURNS_LIST = [7, 7, 7, 7, 7]
MAX_BACKTRACKS_LIST = [0]

# ONE persuasion technique to test (must be one of the supported ones)
# "authority_endorsement", "evidence_based", "expert_endorsement",
# "logical_appeal", "misrepresentation"
PERSUASION_TECHNIQUE = "authority_endorsement"

# All metric names across all tasks (for CSV header)
ALL_METRIC_NAMES = sorted(
    {
        metric["name"]
        for cfg in TASK_CONFIGS.values()
        for metric in cfg.get("likert_metrics", [])
    }
)


# --- Helper: Universal Text Extractor ---
def get_message_content(entry):
    """Safe text extractor for logs and CSV."""
    possible_attrs = ["content", "value", "original_value", "text", "converted_value"]
    for attr in possible_attrs:
        if hasattr(entry, attr):
            val = getattr(entry, attr)
            if val:
                return str(val)
    return str(entry)


# --- Helper: Save Clean Log ---
def save_readable_log(conversation_id, file_path, result, task_name, objective):
    memory = CentralMemory.get_memory_instance()
    entries = memory.get_conversation(conversation_id=conversation_id)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== Experiment Transcript ===\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Objective: {objective}\n")
        f.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        for entry in entries:
            if hasattr(entry, "role") and str(entry.role).lower() == "system":
                continue

            role_display = (
                "ATTACKER"
                if hasattr(entry, "role") and str(entry.role).lower() == "user"
                else "TARGET"
            )
            f.write(f"[{role_display}]\n")
            f.write(f"{get_message_content(entry)}\n")
            f.write("-" * 30 + "\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("=== FINAL RESULT ===\n")
        f.write(f"Outcome: {result.outcome}\n")
        f.write(f"Turns Used: {result.executed_turns}\n")
        f.write(f"Backtracks: {result.backtrack_count}\n")
        f.write("=" * 50 + "\n")


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
        auxiliary_scorers=list(likert_scorers.values()),
    )

    return scoring_config, objective_scorer, likert_scorers


def build_converter_config() -> AttackConverterConfig:
    """
    Creates an AttackConverterConfig that rewrites attacker requests
    using the LLM-based PersuasionConverter from PyRIT.
    """
    # Separate LLM used only for rewriting prompts with persuasion
    converter_llm = OpenAIChatTarget(model_name="gpt-4o-mini")

    persuasion_converter = PersuasionConverter(
        converter_target=converter_llm,
        persuasion_technique=PERSUASION_TECHNIQUE,
    )

    request_converters = PromptConverterConfiguration.from_converters(
        converters=[persuasion_converter]
    )

    return AttackConverterConfig(
        request_converters=request_converters,
    )


async def run_single_experiment(
    max_turns: int,
    max_backtracks: int,
    experiment_id: int,
    output_folder: str,
    task_name: str,
    task_cfg: dict,
    likert_metric_names: list[str],
    objective: str,
    rep_idx: int,
):
    log_filename = (
        f"{task_name}_exp_{experiment_id:03d}_rep{rep_idx}_"
        f"Turns{max_turns}_Backtracks{max_backtracks}.txt"
    )
    full_log_path = os.path.join(output_folder, log_filename)

    print(
        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
        f"Task {task_name} | Rep {rep_idx} | Exp {experiment_id} -> {log_filename}"
    )

    CentralMemory.set_memory_instance(SQLiteMemory())

    # Roles
    victim = OpenAIChatTarget(model_name="gpt-3.5-turbo")
    adversary = OpenAIChatTarget(model_name="gpt-4o-mini")
    judge_llm = OpenAIChatTarget(model_name="gpt-4o-mini")

    scoring_config, tf_scorer, likert_scorers = build_scorers_for_task(
        task_cfg,
        judge_llm,
    )

    converter_config = build_converter_config()

    crescendo = CrescendoAttack(
        objective_target=victim,
        attack_adversarial_config=AttackAdversarialConfig(target=adversary),
        attack_scoring_config=scoring_config,
        attack_converter_config=converter_config,
        max_turns=max_turns,
        max_backtracks=max_backtracks,
    )

    try:
        result = await crescendo.execute_async(objective=objective)

        # --- Per-turn Likert float scores (post-hoc analysis for ALL metrics) ---
        memory = CentralMemory.get_memory_instance()
        all_pieces = memory.get_message_pieces(conversation_id=result.conversation_id)

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

                vals = [float(s.score_value) for s in ordered_scores]
                per_turn_scores[metric_name] = vals
                if vals:
                    last_scores[metric_name] = vals[-1]

        # Save human-readable log
        save_readable_log(
            result.conversation_id,
            full_log_path,
            result,
            task_name=task_name,
            objective=objective,
        )

        preview_text = "No response"
        if result.last_response:
            preview_text = get_message_content(result.last_response)

        row = {
            "id": experiment_id,
            "task": task_name,
            "rep": rep_idx,
            "persuasion_technique": PERSUASION_TECHNIQUE,
            "max_turns": max_turns,
            "max_backtracks": max_backtracks,
            "outcome": str(result.outcome),
            "actual_turns": result.executed_turns,
            "actual_backtracks": result.backtrack_count,
            "response_preview": preview_text[:100].replace("\n", " "),
        }

        # Add metrics: for ALL_METRIC_NAMES, fill values or leave empty
        for metric_name in ALL_METRIC_NAMES:
            if metric_name in likert_metric_names:
                vals = per_turn_scores.get(metric_name, [])
                last_val = last_scores.get(metric_name, None)
                per_turn_str = "|".join(f"{v:.3f}" for v in vals) if vals else ""
                last_str = f"{last_val:.3f}" if last_val is not None else ""
            else:
                per_turn_str = ""
                last_str = ""

            row[f"{metric_name}_per_turn"] = per_turn_str
            row[f"{metric_name}_last"] = last_str

        return row

    except Exception as e:
        with open(full_log_path, "w", encoding="utf-8") as f:
            f.write(f"!!! CRASH REPORT !!!\nError: {str(e)}")

        # On error, still return a row with empty metric fields
        row = {
            "id": experiment_id,
            "task": task_name,
            "rep": rep_idx,
            "persuasion_technique": PERSUASION_TECHNIQUE,
            "max_turns": max_turns,
            "max_backtracks": max_backtracks,
            "outcome": "ERROR",
            "actual_turns": 0,
            "actual_backtracks": 0,
            "response_preview": f"Error: {str(e)}",
        }
        for metric_name in ALL_METRIC_NAMES:
            row[f"{metric_name}_per_turn"] = ""
            row[f"{metric_name}_last"] = ""
        return row


async def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_folder_name = f"batch_{timestamp}"
    output_folder = os.path.join("experiment_results", batch_folder_name)

    os.makedirs(output_folder, exist_ok=True)
    CentralMemory.set_memory_instance(SQLiteMemory())
    logging.getLogger().setLevel(logging.WARNING)

    print("=== Batch Starting ===")
    print(f"Tasks: {', '.join(TASK_CONFIGS.keys())}")
    print(f"Persuasion technique: {PERSUASION_TECHNIQUE}")
    print(f"Results folder: {output_folder}")

    results = []
    exp_id = 1

    TASKS_TO_RUN = ["GovHack"]  # just this for debugging

    # Loop over selected tasks
    for task_name, task_cfg in TASK_CONFIGS.items():
        if task_name not in TASKS_TO_RUN:
            continue

        objective = task_cfg["objective"]
        likert_metric_names = [m["name"] for m in task_cfg.get("likert_metrics", [])]

        # Replications for each task
        for rep_idx, turns in enumerate(MAX_TURNS_LIST, start=1):
            for backtracks in MAX_BACKTRACKS_LIST:
                row = await run_single_experiment(
                    turns,
                    backtracks,
                    exp_id,
                    output_folder,
                    task_name,
                    task_cfg,
                    likert_metric_names,
                    objective,
                    rep_idx,
                )
                results.append(row)
                exp_id += 1
                await asyncio.sleep(5)

    # Build CSV header: base + all metric columns
    base_fields = [
        "id",
        "task",
        "rep",
        "persuasion_technique",
        "max_turns",
        "max_backtracks",
        "outcome",
        "actual_turns",
        "actual_backtracks",
        "response_preview",
    ]
    metric_fields = []
    for metric_name in ALL_METRIC_NAMES:
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
