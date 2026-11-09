"""Popularity baseline using the exact Article candidates and target splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def evaluate(bundle: Path) -> dict:
    root = bundle / "expressgnn"
    entities = [line for line in (root / "entities.txt").read_text(encoding="utf-8").splitlines()
                if line.startswith("article:")]
    train = [line.split("\t") for line in (root / "train.txt").read_text(encoding="utf-8").splitlines() if line]
    valid = [line.split("\t") for line in (root / "valid.txt").read_text(encoding="utf-8").splitlines() if line]
    test = [line.split("\t") for line in (root / "test.txt").read_text(encoding="utf-8").splitlines() if line]
    outgoing = Counter(row[0] for row in train)
    incoming = Counter(row[2] for row in train)
    ranks = []
    for head, _, tail in test:
        tail_scores = sorted(((incoming[candidate], candidate) for candidate in entities), reverse=True)
        head_scores = sorted(((outgoing[candidate], candidate) for candidate in entities), reverse=True)
        ranks.append(1 + next(i for i, (_, candidate) in enumerate(tail_scores) if candidate == tail))
        ranks.append(1 + next(i for i, (_, candidate) in enumerate(head_scores) if candidate == head))
    mrr = sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0
    hits10 = sum(rank <= 10 for rank in ranks) / len(ranks) if ranks else 0.0
    result = {
        "baseline": "training relatedTo degree popularity",
        "candidate_count": len(entities),
        "train_queries": len(train),
        "validation_queries": len(valid),
        "test_queries": len(test),
        "directional_queries": len(ranks),
        "mrr": mrr,
        "hits_at_10": hits10,
    }
    (bundle / "baseline_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    print(json.dumps(evaluate(parser.parse_args().bundle), indent=2))


if __name__ == "__main__":
    main()
