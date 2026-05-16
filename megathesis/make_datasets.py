import argparse
from pathlib import Path

import torch

from .io import ROOT, load_group, resolve_device
from .moves import random_walks


def parse_depths(text):
    return [int(x) for x in str(text).replace(" ", "").split(",") if x]


def make_split(target, moves, inverse, depths, per_depth, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    states = []
    labels = []
    last = []
    for depth in depths:
        d = torch.full((per_depth,), int(depth), dtype=torch.long, device=device)
        s, m = random_walks(target, moves, inverse, d, gen)
        states.append(s.cpu())
        labels.append(d.cpu())
        last.append(m.cpu())
    states = torch.cat(states, dim=0)
    labels = torch.cat(labels, dim=0)
    last = torch.cat(last, dim=0)
    perm = torch.randperm(states.size(0), generator=torch.Generator().manual_seed(int(seed) + 1009))
    return {"states": states[perm], "depths": labels[perm], "last_moves": last[perm]}


def save_split(path, payload, args, move_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.update(
        {
            "metric": args.metric.upper(),
            "group_id": int(args.group_id),
            "target_id": int(args.target_id),
            "move_names": move_names,
        }
    )
    torch.save(payload, path)
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", type=int, default=900)
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--metric", choices=("UTM", "FTM"), default="UTM")
    parser.add_argument("--buckets", default="5,10,15,20,30,40")
    parser.add_argument("--train-per-depth", type=int, default=20000)
    parser.add_argument("--val-per-depth", type=int, default=3000)
    parser.add_argument("--test-per-depth", type=int, default=3000)
    parser.add_argument("--search-per-depth", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    moves, names, inverse, target = load_group(args.group_id, args.target_id, args.metric, device)
    depths = parse_depths(args.buckets)
    out = Path(args.out_dir) if args.out_dir else ROOT / "datasets" / args.metric.lower()

    specs = [
        ("state_train.pt", args.train_per_depth, args.seed + 11),
        ("state_val.pt", args.val_per_depth, args.seed + 23),
        ("state_test.pt", args.test_per_depth, args.seed + 37),
        ("search_test.pt", args.search_per_depth, args.seed + 51),
    ]
    for name, count, seed in specs:
        save_split(out / name, make_split(target, moves, inverse, depths, count, seed, device), args, names)


if __name__ == "__main__":
    main()
