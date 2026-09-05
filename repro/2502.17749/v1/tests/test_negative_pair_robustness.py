from __future__ import annotations

import hashlib
import json

import numpy as np


def _cache():
    from lpcode_v1.t3 import EnhancedFeatureCache

    origins = [f"origin-{index}.c" for index in range(10)]
    sources = (
        "gpt3.5",
        "gemini-pro",
        "wizardcoder:33b-v1.1",
        "deepseek-coder:33b-instruct",
    )
    rows = [(origin, source) for origin in origins for source in sources]
    count = len(rows)
    origin_values = np.asarray([origin for origin, _source in rows], dtype=str)
    source_values = np.asarray([source for _origin, source in rows], dtype=str)
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return EnhancedFeatureCache(
        language="c",
        human=np.arange(count * 28, dtype=float).reshape(count, 28),
        llm=np.arange(count * 28, dtype=float).reshape(count, 28) + 0.5,
        labels=np.ones(count, dtype=np.int64),
        source_ids=origin_values.copy(),
        human_origin_ids=origin_values.copy(),
        candidate_origin_ids=origin_values.copy(),
        human_code_sha256=np.asarray([digest(f"h:{x}") for x in origin_values]),
        candidate_code_sha256=np.asarray([digest(f"c:{x}:{s}") for x, s in rows]),
        llm_sources=source_values,
        row_sha256=np.asarray([digest(f"row:{x}:{s}") for x, s in rows]),
        human_parse_ok=np.ones(count, dtype=bool),
        llm_parse_ok=np.ones(count, dtype=bool),
        human_backends=np.asarray(["test"] * count),
        llm_backends=np.asarray(["test"] * count),
        human_fallback_reasons=np.asarray([""] * count),
        llm_fallback_reasons=np.asarray([""] * count),
    )


def test_pairing_audit_confirms_fixed_positives_and_isolation() -> None:
    from lpcode_v1.negative_pair_robustness import pairing_audit
    from lpcode_v1.t3 import build_t1_pair_splits

    cache = _cache()
    variants = {
        mode: build_t1_pair_splits(cache, n_splits=5, seed=42, negative_pair_mode=mode)
        for mode in ("current", "random", "hard")
    }
    audit = pairing_audit(variants, classifier_config={"model": "xgb"}, feature_hash="f" * 64)

    assert audit["pass"] is True
    assert audit["same_positive_pairs"] is True
    assert audit["classifier_config_sha256"] == hashlib.sha256(
        json.dumps({"model": "xgb"}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert audit["feature_extraction_sha256"] == "f" * 64
    assert all(item["positives_equal_negatives"] for item in audit["variants"].values())
