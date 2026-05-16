import argparse
from pathlib import Path

import torch

from .io import ROOT, build_model_from_checkpoint, latest_checkpoint, load_group, resolve_device, save_json
from .search import BeamSolver, summarize


def parse_depths(text):
    if not text:
        return None
    return {int(x) for x in str(text).replace(" ", "").split(",") if x}


def load_states(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        data = {"states": data.unsqueeze(0) if data.ndim == 1 else data, "depths": torch.full((1 if data.ndim == 1 else data.size(0),), -1)}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--kind", choices=("state", "neighbour"), default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--metric", choices=("UTM", "FTM"), default="UTM")
    parser.add_argument("--depths", default="")
    parser.add_argument("--tests", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    kind = args.kind or "state"
    checkpoint = args.checkpoint or latest_checkpoint(kind, args.metric)
    model, payload = build_model_from_checkpoint(checkpoint, device)
    kind = payload.get("kind", kind)
    data_path = Path(args.data_path) if args.data_path else ROOT / "datasets" / payload.get("metric", "UTM").lower() / "search_test.pt"
    data = load_states(data_path)
    metric = data.get("metric", payload.get("metric", "UTM"))
    if payload.get("metric") and payload.get("metric") != metric:
        raise ValueError(f"checkpoint metric {payload.get('metric')} does not match dataset metric {metric}")
    group_id = data.get("group_id", 900)
    target_id = data.get("target_id", 0)
    moves, names, inverse, target = load_group(group_id, target_id, metric, device)
    states = data["states"]
    depths = data.get("depths", torch.full((states.size(0),), -1)).long()
    keep_depths = parse_depths(args.depths)
    if keep_depths is not None:
        mask = torch.tensor([int(d) in keep_depths for d in depths.tolist()], dtype=torch.bool)
        states = states[mask]
        depths = depths[mask]
    if args.tests > 0:
        states = states[:args.tests]
        depths = depths[:args.tests]

    solver = BeamSolver(model, kind, moves, names, inverse, target, device, args.batch_size)
    results = []
    for i, state in enumerate(states):
        result = solver.solve(state, args.beam_width, args.max_depth)
        result["test_num"] = i
        result["scramble_depth"] = int(depths[i].item())
        result["move_names"] = None if result["moves"] is None else [names[m] for m in result["moves"]]
        result["verified"] = solver.verify(state, result["moves"])
        results.append(result)
        print({k: result[k] for k in ("test_num", "scramble_depth", "success", "length", "time", "reason")})

    by_depth = {}
    for depth in sorted(set(int(x.item()) for x in depths)):
        subset = [r for r in results if r["scramble_depth"] == depth]
        by_depth[str(depth)] = summarize(subset)
    payload_out = {
        "checkpoint": str(checkpoint),
        "kind": kind,
        "metric": metric,
        "beam_width": args.beam_width,
        "max_depth": args.max_depth,
        "summary": summarize(results),
        "by_depth": by_depth,
        "results": results,
    }
    out = Path(args.out) if args.out else ROOT / "logs" / f"search_{kind}_{metric.lower()}_B{args.beam_width}.json"
    save_json(out, payload_out)
    print(payload_out["summary"])


if __name__ == "__main__":
    main()
