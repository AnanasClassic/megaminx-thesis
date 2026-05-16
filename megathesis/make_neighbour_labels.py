import argparse
from pathlib import Path

import torch

from .io import ROOT, build_model_from_checkpoint, load_group, resolve_device
from .model import batch_predict
from .moves import all_children


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default="")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--teacher-batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    state_path = Path(args.state_path) if args.state_path else ROOT / "datasets" / "utm" / "state_train.pt"
    data = torch.load(state_path, map_location="cpu", weights_only=False)
    moves, names, _, target = load_group(data["group_id"], data["target_id"], data["metric"], device)
    teacher, _ = build_model_from_checkpoint(args.teacher, device)
    states = data["states"]
    labels = []
    with torch.inference_mode():
        for start in range(0, states.size(0), args.batch_size):
            batch = states[start:start + args.batch_size].to(device)
            children = all_children(batch, moves)
            flat = children.reshape(-1, children.size(-1))
            pred = batch_predict(teacher, flat, device, args.teacher_batch_size).view(batch.size(0), moves.size(0))
            solved = (children == target.view(1, 1, -1)).all(dim=2)
            labels.append(torch.where(solved, pred.new_zeros(()), pred).cpu().half())
            print(start + batch.size(0), states.size(0))
    out = Path(args.out) if args.out else state_path.with_name("neigh_val.pt")
    payload = {k: data[k] for k in ("states", "depths", "last_moves", "metric", "group_id", "target_id")}
    payload.update({"labels": torch.cat(labels, dim=0), "move_names": names, "teacher": str(args.teacher)})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(out)


if __name__ == "__main__":
    main()
