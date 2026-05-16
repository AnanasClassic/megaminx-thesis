import argparse
import time
from pathlib import Path

import torch

from .io import ROOT, build_model_from_checkpoint, load_group, resolve_device, save_json
from .metrics import ranking_against_move, regression
from .model import batch_predict
from .moves import all_children


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    data_path = Path(args.data_path) if args.data_path else ROOT / "datasets" / "utm" / "state_test.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    model, payload = build_model_from_checkpoint(args.checkpoint, device)
    moves, _, inverse, target = load_group(data["group_id"], data["target_id"], data["metric"], device)
    states = data["states"]
    depths = data["depths"].float()

    started = time.perf_counter()
    pred = batch_predict(model, states, device, args.batch_size).view(-1).cpu()
    scalar_time = time.perf_counter() - started

    scores = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, states.size(0), max(1, args.batch_size // moves.size(0))):
            batch = states[start:start + max(1, args.batch_size // moves.size(0))].to(device)
            children = all_children(batch, moves).reshape(-1, target.numel())
            scores.append(batch_predict(model, children, device, args.batch_size).view(batch.size(0), moves.size(0)).cpu())
    rank_time = time.perf_counter() - started

    last = data["last_moves"].long()
    good = torch.where(last >= 0, inverse.cpu()[last.clamp_min(0)], torch.full_like(last, -1))
    result = {
        "checkpoint": str(args.checkpoint),
        "data": str(data_path),
        "kind": payload.get("kind"),
        "metric": data["metric"],
        "samples": int(states.size(0)),
        **regression(pred, depths),
        "state_inputs_per_sec": states.size(0) / max(scalar_time, 1e-9),
        "local_ranking_time": rank_time,
        "local_moves_per_sec": states.size(0) * moves.size(0) / max(rank_time, 1e-9),
        **{f"good_move_{k}": v for k, v in ranking_against_move(torch.cat(scores), good).items()},
    }
    out = Path(args.out) if args.out else ROOT / "logs" / f"eval_state_{Path(args.checkpoint).stem}.json"
    save_json(out, result)
    print(result)


if __name__ == "__main__":
    main()
