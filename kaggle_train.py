"""Kaggle entry point for the official ExpressGNN implementation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expressgnn", type=Path, default=Path("ExpressGNN"))
    parser.add_argument("--experiment", type=Path, default=Path("outputs/expressgnn"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
    "--trans",
    type=int,
    choices=[0, 1],
    default=0,
    )
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.expressgnn = args.expressgnn.resolve()
    args.experiment = args.experiment.resolve()
    data_root = args.bundle / "expressgnn"
    required = ("entities.txt", "relations.txt", "facts.txt", "train.txt", "valid.txt", "test.txt", "rules.txt")
    missing = [name for name in required if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Bundle is missing: {', '.join(missing)}")
    stats_path = args.bundle / "kg_statistics.json"
    if stats_path.is_file():
        print(json.dumps(json.loads(stats_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    args.experiment.mkdir(parents=True, exist_ok=True)
    (args.experiment / "run_config.json").write_text(json.dumps({
        "bundle": str(args.bundle), "expressgnn": str(args.expressgnn),
        "epochs": args.epochs, "num_batches": args.num_batches,
        "batchsize": args.batchsize, "device": args.device,
        "seed": args.seed,
    }, indent=2), encoding="utf-8")
    command = [
        sys.executable, "-m", "main.train",
        "-data_root", str(data_root),
        "-rule_filename", "rules.txt",
        "-load_method", "1",
        "-use_gcn", "1",
        "-num_epochs", str(args.epochs),
        "-num_batches", str(args.num_batches),
        "-batchsize", str(args.batchsize),
        "-seed", str(args.seed),
        "-device", args.device,
        "-exp_folder", str(args.experiment.parent),
        "-exp_name", args.experiment.name,
        "-trans", str(args.trans),
    ]
    completed = subprocess.run(command, cwd=args.expressgnn, check=False,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (args.experiment / "training.log").write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    run_dirs = [path for path in args.experiment.iterdir() if path.is_dir()] if args.experiment.is_dir() else []
    run_dir = max(run_dirs, key=lambda path: path.stat().st_mtime) if run_dirs else args.experiment
    expected = (run_dir / "options.txt",)
    missing_outputs = [str(path) for path in expected if not path.is_file()]
    saved_model = run_dir / "saved_model"
    performance_reports = list(run_dir.glob("performance_hits_*.txt"))
    if missing_outputs or not saved_model.is_dir() or not performance_reports:
        raise RuntimeError("Training completed without expected logs/checkpoints: " +
                           ", ".join(missing_outputs or [str(saved_model), "performance report"]))
    print(f"Verified ExpressGNN outputs in {run_dir}")
    metrics = {}
    for line in performance_reports[0].read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in {"mmr", "hits"}:
            metrics["mrr" if parts[0] == "mmr" else "hits_at_10"] = float(parts[1])
    (args.experiment / "evaluation_results.json").write_text(json.dumps({
        "run_directory": str(run_dir),
        "performance_report": str(performance_reports[0]),
        **metrics,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
