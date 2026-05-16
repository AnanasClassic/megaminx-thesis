import argparse
import time
from pathlib import Path

import torch

from .io import ROOT, build_model_from_checkpoint, resolve_device, save_json
from .metrics import ranking, regression
from .model import batch_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    data_path = Path(args.data_path) if args.data_path else ROOT / "datasets" / "utm" / "neigh_val.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    model, payload = build_model_from_checkpoint(args.checkpoint, device)
    states = data["states"]
    labels = data["labels"].float()

    started = time.perf_counter()
    pred = batch_predict(model, states, device, args.batch_size).cpu()
    elapsed = time.perf_counter() - started
    result = {
        "checkpoint": str(args.checkpoint),
        "data": str(data_path),
        "kind": payload.get("kind"),
        "metric": data["metric"],
        "samples": int(states.size(0)),
        **regression(pred, labels),
        **ranking(pred, labels),
        "state_inputs_per_sec": states.size(0) / max(elapsed, 1e-9),
        "moves_per_sec": states.size(0) * labels.size(1) / max(elapsed, 1e-9),
    }
    out = Path(args.out) if args.out else ROOT / "logs" / f"eval_neighbour_{Path(args.checkpoint).stem}.json"
    save_json(out, result)
    print(result)


if __name__ == "__main__":
    main()
