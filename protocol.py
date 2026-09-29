from __future__ import annotations

import config


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


DEFAULT_DATA_PATH = config.DATA_PATH
DEFAULT_LABEL_COL = config.LABEL_COL
DEFAULT_CONFIG_PATH = "configs/mtam_hg.yaml"
DEFAULT_SYNTHETIC_DATA_PATH = config.SYNTHETIC_DATA_PATH

DEFAULT_SEEDS = list(range(config.SEED, config.SEED + 10))
DEFAULT_SPLIT_METHOD = config.SPLIT_METHOD
DEFAULT_GENERATION_SEED = config.TABDIFF_GENERATION_SEED
DEFAULT_MAIN_OUTPUT_ROOT = "outputs/mtam_hg"

DEFAULT_EPOCHS = config.EPOCHS
DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS = config.SYNTHETIC_PRETRAIN_EPOCHS
DEFAULT_SYNTHETIC_AGENT_EPOCHS = config.SYNTHETIC_AGENT_EPOCHS
DEFAULT_SYNTHETIC_AGENT_LR = config.SYNTHETIC_AGENT_LR
DEFAULT_SYNTHETIC_AGENT_HIDDEN_DIM = config.SYNTHETIC_AGENT_HIDDEN_DIM
DEFAULT_SYNTHETIC_AGENT_ATTENTION_DIM = config.SYNTHETIC_AGENT_ATTENTION_DIM
DEFAULT_SYNTHETIC_AGENT_ATTENTION_HEADS = config.SYNTHETIC_AGENT_ATTENTION_HEADS
DEFAULT_BATCH_SIZE = config.BATCH_SIZE
DEFAULT_LR = config.LR
DEFAULT_WEIGHT_DECAY = config.WEIGHT_DECAY
DEFAULT_DROPOUT = config.DROPOUT
DEFAULT_EARLY_STOPPING_PATIENCE = config.EARLY_STOPPING_PATIENCE

DEFAULT_USE_DYNAMIC_SYNTHETIC_AGENT = config.USE_DYNAMIC_SYNTHETIC_AGENT
DEFAULT_DYNAMIC_SYNTHETIC_REFRESH_EPOCHS = config.DYNAMIC_SYNTHETIC_REFRESH_EPOCHS
DEFAULT_DYNAMIC_SYNTHETIC_WARMUP_EPOCHS = config.DYNAMIC_SYNTHETIC_WARMUP_EPOCHS
DEFAULT_DYNAMIC_SYNTHETIC_USE_SAMPLER = config.DYNAMIC_SYNTHETIC_USE_SAMPLER
DEFAULT_DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT = config.DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT
DEFAULT_DYNAMIC_SYNTHETIC_TOP_RATIO = config.DYNAMIC_SYNTHETIC_TOP_RATIO
DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MIN = config.DYNAMIC_SYNTHETIC_WEIGHT_MIN
DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MAX = config.DYNAMIC_SYNTHETIC_WEIGHT_MAX
DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_BINS = config.DYNAMIC_SYNTHETIC_SCARCITY_BINS
DEFAULT_DYNAMIC_SYNTHETIC_PROCESS_POWER = config.DYNAMIC_SYNTHETIC_PROCESS_POWER
DEFAULT_DYNAMIC_SYNTHETIC_MECHANISM_POWER = config.DYNAMIC_SYNTHETIC_MECHANISM_POWER

DEFAULT_USE_CLUSTER_BALANCE_REWARD = config.USE_CLUSTER_BALANCE_REWARD
DEFAULT_NUM_WORKING_CONDITION_CLUSTERS = config.NUM_WORKING_CONDITION_CLUSTERS
DEFAULT_REWARD_ALPHA_CLUSTER = config.REWARD_ALPHA_CLUSTER

DEFAULT_FINETUNE_BACKBONE_LR = config.FINETUNE_BACKBONE_LR
DEFAULT_FINETUNE_HEAD_LR = config.FINETUNE_HEAD_LR
DEFAULT_FINETUNE_AGENT_LR = config.FINETUNE_AGENT_LR
DEFAULT_FINETUNE_QUALITY_AGENT_LR = config.FINETUNE_QUALITY_AGENT_LR
DEFAULT_FREEZE_FINETUNE_BACKBONE = config.FREEZE_FINETUNE_BACKBONE
DEFAULT_USE_MR_LORA = config.USE_MR_LORA
DEFAULT_MR_LORA_SCOPE = config.MR_LORA_SCOPE
DEFAULT_MR_LORA_RANK_GRAPH = config.MR_LORA_RANK_GRAPH
DEFAULT_MR_LORA_RANK_ROUTING = config.MR_LORA_RANK_ROUTING
DEFAULT_MR_LORA_ALPHA_GRAPH = config.MR_LORA_ALPHA_GRAPH
DEFAULT_MR_LORA_ALPHA_ROUTING = config.MR_LORA_ALPHA_ROUTING
DEFAULT_MR_LORA_DROPOUT = config.MR_LORA_DROPOUT
DEFAULT_MR_LORA_TRAIN_OUTPUT_HEAD = config.MR_LORA_TRAIN_OUTPUT_HEAD
DEFAULT_MAIN_CHECKPOINT_SELECTION_METRIC = config.CHECKPOINT_SELECTION_METRIC
DEFAULT_TABDIFF_NUM_SAMPLES = config.TABDIFF_NUM_SAMPLES


SUPERVISED_MAIN_TRAIN_MODE = "train_with_tabdiff_pretrain"
SUPERVISED_MAIN_EXPERIMENT_NAME = "mtam_hg_paper"

MAIN_PY_MODE_CHOICES = (
    "evaluate",
    "generate_synthetic_tabdiff",
    "pretrain_synthetic",
    SUPERVISED_MAIN_TRAIN_MODE,
)

DYNAMIC_SYNTHETIC_RUNNER_ARG_SPECS = [
    ("use_dynamic_synthetic_agent", "--use_dynamic_synthetic_agent"),
    ("dynamic_synthetic_refresh_epochs", "--dynamic_synthetic_refresh_epochs"),
    ("dynamic_synthetic_warmup_epochs", "--dynamic_synthetic_warmup_epochs"),
    ("dynamic_synthetic_use_sampler", "--dynamic_synthetic_use_sampler"),
    ("dynamic_synthetic_use_loss_weight", "--dynamic_synthetic_use_loss_weight"),
    ("dynamic_synthetic_top_ratio", "--dynamic_synthetic_top_ratio"),
    ("dynamic_synthetic_weight_min", "--dynamic_synthetic_weight_min"),
    ("dynamic_synthetic_weight_max", "--dynamic_synthetic_weight_max"),
    ("dynamic_synthetic_scarcity_bins", "--dynamic_synthetic_scarcity_bins"),
    ("dynamic_synthetic_process_power", "--dynamic_synthetic_process_power"),
    ("dynamic_synthetic_mechanism_power", "--dynamic_synthetic_mechanism_power"),
]

CLUSTER_BALANCE_ARG_SPECS = [
    ("use_cluster_balance_reward", "--use_cluster_balance_reward"),
    ("num_working_condition_clusters", "--num_working_condition_clusters"),
    ("reward_alpha_cluster", "--reward_alpha_cluster"),
]

MAIN_TRAIN_ARG_SPECS = [
    ("dropout", "--dropout"),
    ("agent_dropout", "--agent_dropout"),
    ("synthetic_agent_epochs", "--synthetic_agent_epochs"),
    ("synthetic_agent_lr", "--synthetic_agent_lr"),
    ("synthetic_agent_hidden_dim", "--synthetic_agent_hidden_dim"),
    ("synthetic_agent_attention_dim", "--synthetic_agent_attention_dim"),
    ("synthetic_agent_attention_heads", "--synthetic_agent_attention_heads"),
    ("synthetic_agent_dropout", "--synthetic_agent_dropout"),
    *DYNAMIC_SYNTHETIC_RUNNER_ARG_SPECS,
    ("finetune_backbone_lr", "--finetune_backbone_lr"),
    ("finetune_head_lr", "--finetune_head_lr"),
    ("finetune_agent_lr", "--finetune_agent_lr"),
    ("finetune_quality_agent_lr", "--finetune_quality_agent_lr"),
    ("use_layerwise_finetune_lr", "--use_layerwise_finetune_lr"),
    ("freeze_finetune_backbone", "--freeze_finetune_backbone"),
    ("early_stopping_patience", "--early_stopping_patience"),
    ("checkpoint_selection_metric", "--checkpoint_selection_metric"),
    *CLUSTER_BALANCE_ARG_SPECS,
]

MR_LORA_ARG_SPECS = [
    ("use_mr_lora", "--use_mr_lora"),
    ("mr_lora_scope", "--mr_lora_scope"),
    ("mr_lora_rank_graph", "--mr_lora_rank_graph"),
    ("mr_lora_rank_routing", "--mr_lora_rank_routing"),
    ("mr_lora_alpha_graph", "--mr_lora_alpha_graph"),
    ("mr_lora_alpha_routing", "--mr_lora_alpha_routing"),
    ("mr_lora_dropout", "--mr_lora_dropout"),
    ("mr_lora_train_output_head", "--mr_lora_train_output_head"),
]

BOOL_VALUE_FLAGS = {
    "--dynamic_synthetic_use_sampler",
    "--dynamic_synthetic_use_loss_weight",
}
