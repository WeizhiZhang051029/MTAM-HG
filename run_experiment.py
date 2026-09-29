from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from metrics import REGRESSION_METRIC_NAMES
from pipeline import build_parser as build_pipeline_parser
from protocol import (
    BOOL_VALUE_FLAGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_DYNAMIC_SYNTHETIC_MECHANISM_POWER,
    DEFAULT_DYNAMIC_SYNTHETIC_PROCESS_POWER,
    DEFAULT_DYNAMIC_SYNTHETIC_REFRESH_EPOCHS,
    DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_BINS,
    DEFAULT_DYNAMIC_SYNTHETIC_TOP_RATIO,
    DEFAULT_DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT,
    DEFAULT_DYNAMIC_SYNTHETIC_USE_SAMPLER,
    DEFAULT_DYNAMIC_SYNTHETIC_WARMUP_EPOCHS,
    DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MAX,
    DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MIN,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_FINETUNE_AGENT_LR,
    DEFAULT_FINETUNE_BACKBONE_LR,
    DEFAULT_FINETUNE_HEAD_LR,
    DEFAULT_FINETUNE_QUALITY_AGENT_LR,
    DEFAULT_FREEZE_FINETUNE_BACKBONE,
    DEFAULT_GENERATION_SEED,
    DEFAULT_LABEL_COL,
    DEFAULT_LR,
    DEFAULT_MAIN_CHECKPOINT_SELECTION_METRIC,
    DEFAULT_MAIN_OUTPUT_ROOT,
    DEFAULT_MR_LORA_ALPHA_GRAPH,
    DEFAULT_MR_LORA_ALPHA_ROUTING,
    DEFAULT_MR_LORA_DROPOUT,
    DEFAULT_MR_LORA_RANK_GRAPH,
    DEFAULT_MR_LORA_RANK_ROUTING,
    DEFAULT_MR_LORA_SCOPE,
    DEFAULT_MR_LORA_TRAIN_OUTPUT_HEAD,
    DEFAULT_NUM_WORKING_CONDITION_CLUSTERS,
    DEFAULT_REWARD_ALPHA_CLUSTER,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_METHOD,
    DEFAULT_SYNTHETIC_AGENT_ATTENTION_DIM,
    DEFAULT_SYNTHETIC_AGENT_ATTENTION_HEADS,
    DEFAULT_SYNTHETIC_AGENT_EPOCHS,
    DEFAULT_SYNTHETIC_AGENT_HIDDEN_DIM,
    DEFAULT_SYNTHETIC_AGENT_LR,
    DEFAULT_SYNTHETIC_DATA_PATH,
    DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS,
    DEFAULT_TABDIFF_NUM_SAMPLES,
    DEFAULT_USE_CLUSTER_BALANCE_REWARD,
    DEFAULT_USE_DYNAMIC_SYNTHETIC_AGENT,
    DEFAULT_USE_MR_LORA,
    DEFAULT_WEIGHT_DECAY,
    MAIN_TRAIN_ARG_SPECS,
    MR_LORA_ARG_SPECS,
    SUPERVISED_MAIN_EXPERIMENT_NAME,
    SUPERVISED_MAIN_TRAIN_MODE,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER_SUMMARY_NAME = "run_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = build_pipeline_parser()
    parser.set_defaults(
        config=DEFAULT_CONFIG_PATH,
        data_path=DEFAULT_DATA_PATH,
        label_col=DEFAULT_LABEL_COL,
        synthetic_data_path=DEFAULT_SYNTHETIC_DATA_PATH,
        output_dir="",
        experiment_name="",
        seeds=DEFAULT_SEEDS,
        output_root=DEFAULT_MAIN_OUTPUT_ROOT,
        main_experiment_name="",
        generation_seed=DEFAULT_GENERATION_SEED,
        epochs=DEFAULT_EPOCHS,
        synthetic_pretrain_epochs=DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS,
        synthetic_agent_epochs=DEFAULT_SYNTHETIC_AGENT_EPOCHS,
        synthetic_agent_lr=DEFAULT_SYNTHETIC_AGENT_LR,
        synthetic_agent_hidden_dim=DEFAULT_SYNTHETIC_AGENT_HIDDEN_DIM,
        synthetic_agent_attention_dim=DEFAULT_SYNTHETIC_AGENT_ATTENTION_DIM,
        synthetic_agent_attention_heads=DEFAULT_SYNTHETIC_AGENT_ATTENTION_HEADS,
        dropout=DEFAULT_DROPOUT,
        agent_dropout=DEFAULT_DROPOUT,
        synthetic_agent_dropout=DEFAULT_DROPOUT,
        use_dynamic_synthetic_agent=DEFAULT_USE_DYNAMIC_SYNTHETIC_AGENT,
        dynamic_synthetic_refresh_epochs=DEFAULT_DYNAMIC_SYNTHETIC_REFRESH_EPOCHS,
        dynamic_synthetic_warmup_epochs=DEFAULT_DYNAMIC_SYNTHETIC_WARMUP_EPOCHS,
        dynamic_synthetic_use_sampler=DEFAULT_DYNAMIC_SYNTHETIC_USE_SAMPLER,
        dynamic_synthetic_use_loss_weight=DEFAULT_DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT,
        dynamic_synthetic_top_ratio=DEFAULT_DYNAMIC_SYNTHETIC_TOP_RATIO,
        dynamic_synthetic_weight_min=DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MIN,
        dynamic_synthetic_weight_max=DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MAX,
        dynamic_synthetic_scarcity_bins=DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_BINS,
        dynamic_synthetic_process_power=DEFAULT_DYNAMIC_SYNTHETIC_PROCESS_POWER,
        dynamic_synthetic_mechanism_power=DEFAULT_DYNAMIC_SYNTHETIC_MECHANISM_POWER,
        use_cluster_balance_reward=DEFAULT_USE_CLUSTER_BALANCE_REWARD,
        num_working_condition_clusters=DEFAULT_NUM_WORKING_CONDITION_CLUSTERS,
        reward_alpha_cluster=DEFAULT_REWARD_ALPHA_CLUSTER,
        finetune_backbone_lr=DEFAULT_FINETUNE_BACKBONE_LR,
        finetune_head_lr=DEFAULT_FINETUNE_HEAD_LR,
        finetune_agent_lr=DEFAULT_FINETUNE_AGENT_LR,
        finetune_quality_agent_lr=DEFAULT_FINETUNE_QUALITY_AGENT_LR,
        use_layerwise_finetune_lr=True,
        freeze_finetune_backbone=DEFAULT_FREEZE_FINETUNE_BACKBONE,
        use_mr_lora=DEFAULT_USE_MR_LORA,
        mr_lora_scope=DEFAULT_MR_LORA_SCOPE,
        mr_lora_rank_graph=DEFAULT_MR_LORA_RANK_GRAPH,
        mr_lora_rank_routing=DEFAULT_MR_LORA_RANK_ROUTING,
        mr_lora_alpha_graph=DEFAULT_MR_LORA_ALPHA_GRAPH,
        mr_lora_alpha_routing=DEFAULT_MR_LORA_ALPHA_ROUTING,
        mr_lora_dropout=DEFAULT_MR_LORA_DROPOUT,
        mr_lora_train_output_head=DEFAULT_MR_LORA_TRAIN_OUTPUT_HEAD,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
        checkpoint_selection_metric=DEFAULT_MAIN_CHECKPOINT_SELECTION_METRIC,
        split_method=DEFAULT_SPLIT_METHOD,
        tabdiff_num_samples=None,
        tabdiff_gpu=None,
    )
    parser.add_argument("--output_root", default=DEFAULT_MAIN_OUTPUT_ROOT)
    parser.add_argument("--main_experiment_name", default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--skip_tabdiff_generation",
        action="store_true",
        help="Skip the TabDiff generation phase even when --synthetic_data_path is missing.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Choose one or more distinct run seeds.")
    if args.generation_seed != 0:
        raise ValueError("TabDiff deterministic generation uses seed 0.")
    positive_ints = {
        "epochs": args.epochs,
        "synthetic_pretrain_epochs": args.synthetic_pretrain_epochs,
        "batch_size": args.batch_size,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"--{name} must be positive, got {value}.")
    if args.synthetic_agent_epochs is not None and int(args.synthetic_agent_epochs) <= 0:
        raise ValueError(f"--synthetic_agent_epochs must be positive, got {args.synthetic_agent_epochs}.")
    if args.synthetic_agent_hidden_dim is not None and int(args.synthetic_agent_hidden_dim) <= 0:
        raise ValueError(f"--synthetic_agent_hidden_dim must be positive, got {args.synthetic_agent_hidden_dim}.")
    if args.synthetic_agent_attention_dim is not None and int(args.synthetic_agent_attention_dim) <= 0:
        raise ValueError(
            f"--synthetic_agent_attention_dim must be positive, got {args.synthetic_agent_attention_dim}."
        )
    if args.synthetic_agent_attention_heads is not None and int(args.synthetic_agent_attention_heads) <= 0:
        raise ValueError(
            f"--synthetic_agent_attention_heads must be positive, got {args.synthetic_agent_attention_heads}."
        )
    if args.synthetic_agent_attention_dim is not None and args.synthetic_agent_attention_heads is not None:
        if int(args.synthetic_agent_attention_dim) % int(args.synthetic_agent_attention_heads) != 0:
            raise ValueError(
                "--synthetic_agent_attention_dim must be divisible by "
                "--synthetic_agent_attention_heads."
            )
    if args.tabdiff_num_samples is not None and int(args.tabdiff_num_samples) <= 0:
        raise ValueError(f"--tabdiff_num_samples must be positive, got {args.tabdiff_num_samples}.")
    if int(args.dynamic_synthetic_refresh_epochs) <= 0:
        raise ValueError(
            "--dynamic_synthetic_refresh_epochs must be positive, "
            f"got {args.dynamic_synthetic_refresh_epochs}."
        )
    if int(args.dynamic_synthetic_warmup_epochs) < 0:
        raise ValueError(
            "--dynamic_synthetic_warmup_epochs must be non-negative, "
            f"got {args.dynamic_synthetic_warmup_epochs}."
        )
    if int(args.dynamic_synthetic_scarcity_bins) < 2:
        raise ValueError(
            "--dynamic_synthetic_scarcity_bins must be at least 2, "
            f"got {args.dynamic_synthetic_scarcity_bins}."
        )
    if int(args.num_working_condition_clusters) < 2:
        raise ValueError(
            "--num_working_condition_clusters must be at least 2, "
            f"got {args.num_working_condition_clusters}."
        )
    if float(args.lr) <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if float(args.weight_decay) < 0:
        raise ValueError(f"--weight_decay must be non-negative, got {args.weight_decay}.")
    for name in ("dropout", "agent_dropout", "synthetic_agent_dropout"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}.")
    if int(args.early_stopping_patience) <= 0:
        raise ValueError(f"--early_stopping_patience must be positive, got {args.early_stopping_patience}.")
    finetune_lrs = {
        "finetune_backbone_lr": args.finetune_backbone_lr,
        "finetune_head_lr": args.finetune_head_lr,
        "finetune_agent_lr": args.finetune_agent_lr,
        "finetune_quality_agent_lr": args.finetune_quality_agent_lr,
    }
    for name, value in finetune_lrs.items():
        if value is not None and float(value) <= 0:
            raise ValueError(f"--{name} must be positive, got {value}.")
    bounded = {
        "dynamic_synthetic_top_ratio": args.dynamic_synthetic_top_ratio,
        "dynamic_synthetic_weight_min": args.dynamic_synthetic_weight_min,
        "dynamic_synthetic_process_power": args.dynamic_synthetic_process_power,
    }
    for name, value in bounded.items():
        if value is None:
            continue
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}.")
    if args.dynamic_synthetic_top_ratio is not None and float(args.dynamic_synthetic_top_ratio) <= 0.0:
        raise ValueError(f"--dynamic_synthetic_top_ratio must be in (0, 1], got {args.dynamic_synthetic_top_ratio}.")
    nonnegative_dynamic = {
        "dynamic_synthetic_weight_max": args.dynamic_synthetic_weight_max,
        "dynamic_synthetic_mechanism_power": args.dynamic_synthetic_mechanism_power,
        "reward_alpha_cluster": args.reward_alpha_cluster,
    }
    for name, value in nonnegative_dynamic.items():
        if value is not None and float(value) < 0.0:
            raise ValueError(f"--{name} must be non-negative, got {value}.")
    if float(args.dynamic_synthetic_weight_max) < float(args.dynamic_synthetic_weight_min):
        raise ValueError(
            "--dynamic_synthetic_weight_max must be >= --dynamic_synthetic_weight_min, "
            f"got {args.dynamic_synthetic_weight_max} < {args.dynamic_synthetic_weight_min}."
        )
def append_optional_cli_args(
    cmd: list[str],
    args: argparse.Namespace,
    specs: list[tuple[str, str]] | None = None,
) -> list[str]:

    for attr_name, flag in (specs or ()):
        value = getattr(args, attr_name, None)
        if isinstance(value, bool):
            if flag in BOOL_VALUE_FLAGS:
                cmd.extend([flag, str(value)])
            elif value:
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])
    return cmd

def build_main_train_command(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    synthetic_path: str,
    tabdiff_num_samples: int,
    experiment_name: str | None = None,
    scientific_code_hash: str | None = None,
    generation_protocol_hash: str | None = None,
) -> list[str]:
    resolved_experiment_name = experiment_name or args.main_experiment_name or SUPERVISED_MAIN_EXPERIMENT_NAME
    cmd = [
        sys.executable,
        "pipeline.py",
        "--mode",
        SUPERVISED_MAIN_TRAIN_MODE,
        "--config",
        args.config,
        "--experiment_name",
        resolved_experiment_name,
        "--output_dir",
        str(run_dir),
        "--data_path",
        args.data_path,
        "--label_col",
        args.label_col,
        "--synthetic_data_path",
        synthetic_path,
        "--seed",
        str(seed),
        "--split_seed",
        str(seed),
        "--generation_seed",
        str(args.generation_seed),
        "--epochs",
        str(args.epochs),
        "--tabdiff_num_samples",
        str(tabdiff_num_samples),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--split_method",
        args.split_method,
        "--no_el",
        "--no_laplace",
    ]
    cmd.extend(["--synthetic_pretrain_epochs", str(args.synthetic_pretrain_epochs)])
    cmd = append_optional_cli_args(cmd, args, MAIN_TRAIN_ARG_SPECS)
    if not args.use_dynamic_synthetic_agent:
        cmd.append("--no_dynamic_synthetic_agent")
    if not args.use_cluster_balance_reward:
        cmd.append("--no_cluster_balance_reward")
    if not args.use_layerwise_finetune_lr:
        cmd.append("--no_layerwise_finetune_lr")
    if not args.freeze_finetune_backbone:
        cmd.append("--no_freeze_finetune_backbone")
    if args.use_mr_lora:
        mr_lora_value_specs = [spec for spec in MR_LORA_ARG_SPECS if spec[0] not in {"use_mr_lora", "mr_lora_train_output_head"}]
        cmd.extend(
            [
                "--use_mr_lora",
            ]
        )
        cmd = append_optional_cli_args(cmd, args, mr_lora_value_specs)
        if args.mr_lora_train_output_head:
            cmd.append("--mr_lora_train_output_head")
        else:
            cmd.append("--no_mr_lora_train_output_head")
    else:
        cmd.append("--no_mr_lora")
    if (scientific_code_hash is None) != (generation_protocol_hash is None):
        raise ValueError("Scientific code and generation protocol hashes must be provided together.")
    if scientific_code_hash is not None and generation_protocol_hash is not None:
        cmd.extend(
            [
                "--scientific_code_sha256",
                _sha256_value(scientific_code_hash, "scientific code"),
                "--generation_protocol_sha256",
                _sha256_value(generation_protocol_hash, "generation protocol"),
            ]
        )
    return cmd


def seed_template_path(template: str, seed: int) -> str:
    if "{seed}" in template:
        return template.format(seed=seed)
    path = Path(template)
    return str(path.with_name(f"{path.stem}_seed_{seed}{path.suffix}"))


def _sha256_value(value: object, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"Invalid {field}.")
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from config_loader import load_yaml_config

    parser = build_parser()
    initial, _ = parser.parse_known_args(argv)
    values = load_yaml_config(PROJECT_ROOT / initial.config)
    defaults = dict(values)
    defaults.update(values.get("cli", {}))
    defaults["seeds"] = values.get("model_seeds", DEFAULT_SEEDS)
    defaults["output_root"] = values.get("output_base", DEFAULT_MAIN_OUTPUT_ROOT)
    destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{k: v for k, v in defaults.items() if k in destinations})
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def run_experiments(args: argparse.Namespace) -> None:
    from datetime import datetime

    root = (PROJECT_ROOT / args.output_root).resolve()
    root = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    count = int(args.tabdiff_num_samples or DEFAULT_TABDIFF_NUM_SAMPLES)
    commands: list[list[str]] = []
    results: list[dict[str, object]] = []
    for seed in args.seeds:
        run_dir = root / f"seed_{seed}"
        synthetic = str((PROJECT_ROOT / seed_template_path(args.synthetic_data_path, seed)).resolve())
        if args.skip_tabdiff_generation and not Path(synthetic).is_file():
            raise FileNotFoundError(synthetic)
        command = build_main_train_command(args, seed, run_dir, synthetic, count)
        if args.tabdiff_gpu is not None:
            command.extend(["--tabdiff_gpu", str(args.tabdiff_gpu)])
        if args.skip_tabdiff_generation:
            command.append("--require_existing_synthetic")
        commands.append(command)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        paths = list(run_dir.rglob("metrics.json"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one fresh metrics file in {run_dir}, found {len(paths)}.")
        metrics = json.loads(paths[0].read_text(encoding="utf-8"))
        if metrics.get("Seed") != seed or metrics.get("Split_Seed") != seed:
            raise RuntimeError("Run and split seeds do not match.")
        if not all(math.isfinite(float(metrics[m])) for m in REGRESSION_METRIC_NAMES):
            raise RuntimeError(f"Non-finite evaluation metric for seed {seed}.")
        results.append({"seed": seed, "metrics": metrics, "metrics_path": str(paths[0])})
    summary = {
        "protocol": "per_run_split",
        "feedback_source": "real_training_set",
        "config": load_yaml_config(PROJECT_ROOT / args.config),
        "arguments": vars(args),
        "commands": commands,
        "runs": results,
        "aggregate": {
            metric: {
                "mean": float(np.mean([run["metrics"][metric] for run in results])),
                "std": float(np.std([run["metrics"][metric] for run in results], ddof=1)) if len(results) > 1 else None,
            }
            for metric in REGRESSION_METRIC_NAMES
        } if results else {},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / RUNNER_SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {root}")
    return summary


def main() -> None:
    run_experiments(parse_args())


if __name__ == "__main__":
    main()
