import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import ROOT


def mean_ci(values, bootstraps, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, 0.0
    mean = float(values.mean())
    if values.size < 2 or bootstraps <= 0:
        return mean, 0.0
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(bootstraps, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(sample, [2.5, 97.5])
    return mean, max(mean - lo, hi - mean)


def load_rows(paths, bootstraps):
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        solved = [r for r in data["results"] if r["success"]]
        lengths = [r["length"] for r in solved]
        times = [r["time"] for r in data["results"]]
        flops = [r["model_flops"] for r in data["results"]]
        seed = int(data["beam_width"]) + (0 if data["kind"] == "state" else 100000)
        length, length_ci = mean_ci(lengths, bootstraps, seed)
        mean_time, time_ci = mean_ci(times, bootstraps, seed + 7)
        mean_flops, flops_ci = mean_ci(flops, bootstraps, seed + 13)
        rows.append({
            "path": str(path),
            "kind": data["kind"],
            "metric": data["metric"],
            "beam_width": int(data["beam_width"]),
            "success_rate": data["summary"]["success_rate"],
            "mean_length": length,
            "ci_length": length_ci,
            "mean_time": mean_time,
            "ci_time": time_ci,
            "mean_model_flops": mean_flops,
            "ci_model_flops": flops_ci,
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def by_kind(rows):
    for kind in ("state", "neighbour"):
        cur = sorted([r for r in rows if r["kind"] == kind], key=lambda r: r["beam_width"])
        if cur:
            yield kind, cur


def panel(ax, rows, x, y, yerr, title, xlabel, ylabel, xlog=True, ylog=False):
    labels = {"state": "State Model", "neighbour": "Neighbour Model"}
    for kind, cur in by_kind(rows):
        ax.errorbar([r[x] for r in cur], [r[y] for r in cur], yerr=[r[yerr] for r in cur], marker="o", capsize=3, label=labels[kind])
    if xlog:
        ax.set_xscale("log", base=2 if x == "beam_width" else 10)
    if ylog:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.45)
    ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--metric", choices=("UTM", "FTM"), default="UTM")
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--out", default="")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths] or sorted((ROOT / "logs").glob(f"beam_sweep_*_{args.metric.lower()}_B*.json"))
    rows = load_rows(paths, args.bootstraps)
    if not rows:
        raise SystemExit("no sweep logs found")
    csv_path = Path(args.csv) if args.csv else ROOT / "logs" / f"beam_sweep_{args.metric.lower()}.csv"
    write_csv(csv_path, rows)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    panel(axes[0, 0], rows, "beam_width", "mean_length", "ci_length", "Average Solution Length vs Beam Width", "Beam width", "Average solution length")
    panel(axes[0, 1], rows, "mean_time", "mean_length", "ci_length", "Average Solution Length vs Average Time", "Average time (sec)", "Average solution length")
    panel(axes[1, 0], rows, "beam_width", "mean_time", "ci_time", "Average Time vs Beam Width", "Beam width", "Average time (sec)", ylog=True)
    panel(axes[1, 1], rows, "mean_model_flops", "mean_length", "ci_length", "Average Solution Length vs Average Model FLOPs", "Average model FLOPs", "Average solution length")
    out = Path(args.out) if args.out else ROOT / "logs" / f"beam_sweep_{args.metric.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)
    print(csv_path)


if __name__ == "__main__":
    main()
