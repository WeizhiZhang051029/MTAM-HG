from __future__ import annotations

import numpy as np
import torch

import config


def build_node_type_ids(node_names: list[str] | None = None) -> torch.LongTensor:
    node_names = node_names or config.active_node_names()
    ids = [config.NODE_TYPE_TO_ID[config.NODE_TYPE_MAP[name]] for name in node_names]
    return torch.tensor(ids, dtype=torch.long)


def build_stage_ids(node_names: list[str] | None = None) -> torch.LongTensor:
    node_names = node_names or config.active_node_names()
    stage_lookup: dict[str, int] = {}
    for stage_name, names in config.STAGE_NODE_MAP.items():
        stage_id = config.STAGE_TO_ID[stage_name]
        for name in names:
            stage_lookup[name] = stage_id
    stage_lookup.setdefault("EL", config.STAGE_TO_ID[config.QUALITY_STAGE_NAME])
    return torch.tensor([stage_lookup[name] for name in node_names], dtype=torch.long)


def build_process_order_ids(node_names: list[str] | None = None) -> torch.LongTensor:

    node_names = node_names or config.active_node_names()
    order_lookup: dict[str, int] = {}
    for order_name, names in config.PROCESS_ORDER_NODE_MAP.items():
        order_id = config.PROCESS_ORDER_TO_ID[order_name]
        for name in names:
            order_lookup[name] = order_id
    fallback = config.PROCESS_ORDER_TO_ID["order11_shear_coiler_quality"]
    return torch.tensor([order_lookup.get(name, fallback) for name in node_names], dtype=torch.long)


def _add_edges(A: np.ndarray, node_to_idx: dict[str, int], sources: list[str], targets: list[str], weight: float) -> None:
    for src in sources:
        for dst in targets:
            if src in node_to_idx and dst in node_to_idx:
                A[node_to_idx[dst], node_to_idx[src]] = max(A[node_to_idx[dst], node_to_idx[src]], weight)


def build_mechanistic_prior_graph(
    node_names: list[str] | None = None,
    same_type_weight: float = 0.08,
    self_loop_weight: float = 1.0,
) -> np.ndarray:

    node_names = node_names or config.active_node_names()
    n = len(node_names)
    node_to_idx = {name: idx for idx, name in enumerate(node_names)}
    A = np.zeros((n, n), dtype=np.float32)

    groups = config.MECHANISTIC_NODE_GROUPS
    operating = groups["operating"]
    procedure = groups["procedure"]
    conditional = groups["conditional"]
    composition = groups["composition"]

    # Representative relations from the CAPL mechanistic-prior table.
    relations = config.MECHANISTIC_RELATION_GROUPS
    thermal_stage_chain = relations["thermal_stage_chain"]
    hot_history = relations["hot_history"]
    deformation_targets = relations["deformation_targets"]
    cooling_variables = relations["cooling_variables"]
    geometry_nodes = relations["geometry_nodes"]

    _add_edges(A, node_to_idx, operating, procedure, 1.00)
    for source_group, target_group in zip(thermal_stage_chain, thermal_stage_chain[1:]):
        _add_edges(A, node_to_idx, [source_group], [target_group], 0.90)
    _add_edges(A, node_to_idx, hot_history, deformation_targets, 0.90)
    _add_edges(A, node_to_idx, cooling_variables, geometry_nodes, 0.90)
    _add_edges(A, node_to_idx, geometry_nodes, ["RF"], 0.75)
    _add_edges(A, node_to_idx, composition, conditional, 1.00)

    # Light within-type connectivity preserves the heterogeneous node classes.
    for type_name in config.NODE_TYPES:
        type_nodes = [name for name in node_names if config.NODE_TYPE_MAP[name] == type_name]
        _add_edges(A, node_to_idx, type_nodes, type_nodes, same_type_weight)

    np.fill_diagonal(A, self_loop_weight)
    return A


def build_relation_templates(node_names: list[str] | None = None) -> tuple[torch.FloatTensor, list[tuple[str, str]]]:

    node_names = node_names or config.active_node_names()
    type_ids = build_node_type_ids(node_names)
    templates = []
    relation_names: list[tuple[str, str]] = []
    for src_type in config.NODE_TYPES:
        src_id = config.NODE_TYPE_TO_ID[src_type]
        for dst_type in config.NODE_TYPES:
            dst_id = config.NODE_TYPE_TO_ID[dst_type]
            src_mask = type_ids == src_id
            dst_mask = type_ids == dst_id
            template = torch.outer(dst_mask.float(), src_mask.float())
            templates.append(template)
            relation_names.append((src_type, dst_type))
    return torch.stack(templates, dim=0), relation_names


def row_normalize_adjacency(A: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:

    return A / (A.sum(dim=-1, keepdim=True) + eps)
