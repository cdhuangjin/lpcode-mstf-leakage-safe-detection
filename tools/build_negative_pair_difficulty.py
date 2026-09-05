"""Generate prediction-independent pair-difficulty summaries for N0/N1/N2."""

from __future__ import annotations

import csv
import ast
import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "repro" / "2502.17749" / "v1"
sys.path.insert(0, str(REPRO))

from lpcode_v1.negative_pair_robustness import LANGUAGES, MODES, N_SPLITS, SEEDS
from lpcode_v1.t1_strict import _select_positive_bank
from lpcode_v1.t3 import build_t1_pair_splits, load_or_build_enhanced_cache
from lpcode_v1.features_enhanced import _normalize_parser_source, _tree_sitter_parser


@lru_cache(maxsize=None)
def _tokens(code: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z_]\w*|\d+|\S", code))


@lru_cache(maxsize=None)
def _nonempty_lines(code: str) -> int:
    return sum(bool(line.strip()) for line in code.splitlines())


def _ratio(left: int, right: int) -> float:
    return min(left, right) / max(left, right) if left and right else 0.0


@lru_cache(maxsize=None)
def _ast_node_count(language: str, code: str) -> int:
    """Count parsed AST nodes without using labels, model scores or MSTF output."""

    try:
        if language == "py":
            return sum(1 for _ in ast.walk(ast.parse(code)))
        root = _tree_sitter_parser(language).parse(
            _normalize_parser_source(code).encode("utf-8")
        ).root_node
        pending, count = [root], 0
        while pending:
            node = pending.pop()
            count += 1
            pending.extend(node.children)
        return count
    except (SyntaxError, UnicodeError, ValueError):
        return 0


def main() -> None:
    output = ROOT / "results" / "negative_pair_robustness" / "pair_difficulty_summary.csv"
    cache_root = ROOT / "results" / "01_transition_test_strict_origins" / "cache"
    values: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for language in LANGUAGES:
        path = ROOT / "repro" / "2502.17749" / "code" / "experiment" / "task1" / "dataset" / f"{language}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        positive = {
            (str(row["file_name"]), str(row["paraphrased_by"])): (str(row["human_src"]), str(row["llm_src"]))
            for row in rows
            if int(row["label"]) == 1
        }
        cache = load_or_build_enhanced_cache(language, path, cache_root, cache_root)
        cache = _select_positive_bank(cache, None)
        # Difficulty is descriptive only.  One frozen seed's five test folds
        # cover every origin once, avoiding threefold duplicate distributions.
        for seed in (SEEDS[0],):
            for mode in MODES:
                for split in build_t1_pair_splits(cache, language=language, n_splits=N_SPLITS, seed=seed, negative_pair_mode=mode):
                    # Fold-local test partitions cover each origin once per seed.
                    # Using them avoids counting the same origin repeatedly across
                    # the overlapping training folds.
                    for pairs in (split.test_pairs,):
                        for pair in pairs:
                            human, _ = positive[(pair.human_origin_id, pair.llm_source)]
                            _, candidate = positive[(pair.candidate_origin_id, pair.llm_source)]
                            human_tokens, candidate_tokens = _tokens(human), _tokens(candidate)
                            union = human_tokens | candidate_tokens
                            metrics = {
                                "loc_ratio": _ratio(_nonempty_lines(human), _nonempty_lines(candidate)),
                                "token_ratio": _ratio(len(human_tokens), len(candidate_tokens)),
                                "token_jaccard": len(human_tokens & candidate_tokens) / len(union) if union else 1.0,
                                "ast_node_ratio": _ratio(_ast_node_count(language, human), _ast_node_count(language, candidate)),
                                "endpoint_style_distance": float(np.linalg.norm(cache.human[pair.human_positive_row_idx] - cache.llm[pair.candidate_positive_row_idx])),
                            }
                            values[(mode, "positive" if pair.label == 1 else "negative")].append(metrics)
    rows_out: list[dict[str, object]] = []
    for (mode, label), records in sorted(values.items()):
        for metric in ("loc_ratio", "token_ratio", "token_jaccard", "ast_node_ratio", "endpoint_style_distance"):
            metric_values = np.asarray([record[metric] for record in records], dtype=float)
            rows_out.append({"negative_pairing": mode, "pair_label": label, "metric": metric, "mean": float(np.mean(metric_values)), "std": float(np.std(metric_values, ddof=1)), "n_pairs": len(metric_values)})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)


if __name__ == "__main__":
    main()
