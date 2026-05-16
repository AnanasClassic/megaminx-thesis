import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .io import ROOT, load_group, resolve_device, run_id as make_run_id
from .metrics import regression
from .model import Net
from .moves import random_walks
from .train_utils import append_csv, batches, parse_depths, save_checkpoint


def eval_model(model, states, depths, batch_size, device):
    model.eval()
    pred = []
    with torch.inference_mode():
        for idx in batches(states.size(0), batch_size, False, device, 0):
            pred.append(model(states.index_select(0, idx).to(device)).float().cpu())
    return regression(torch.cat(pred), depths.float())


def epoch_depths(depth_values, count, generator):
    per_depth = max(count // depth_values.numel(), 1)
    y = depth_values.repeat_interleave(per_depth)
    return y[torch.randperm(y.numel(), generator=generator, device=y.device)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="")
    parser.add_argument("--val-path", default="")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--depths", default="")
    parser.add_argument("--buckets", default="")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hd1", type=int, default=1536)
    parser.add_argument("--hd2", type=int, default=512)
    parser.add_argument("--nrd", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="state")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    train_path = Path(args.train_path) if args.train_path else ROOT / "datasets" / "utm" / "state_train.pt"
    val_path = Path(args.val_path) if args.val_path else train_path.with_name("state_val.pt")
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    val = torch.load(val_path, map_location="cpu", weights_only=False)
    metric = train["metric"]
    moves, _, inverse, target = load_group(train["group_id"], train["target_id"], metric, device)
    num_classes = int(target.max().item() - min(0, int(target.min().item())) + 1)
    z_add = -int(target.min().item()) if int(target.min().item()) < 0 else 0
    model = Net(target.numel(), num_classes, 1, args.hd1, args.hd2, args.nrd, args.dropout, z_add).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_id = make_run_id()
    prefix = f"{args.run_name}_{metric.lower()}_{run_id}"
    log_path = ROOT / "logs" / f"{prefix}.csv"
    best = math.inf
    best_path = ROOT / "checkpoints" / f"{prefix}_best.pt"
    final_path = ROOT / "checkpoints" / f"{prefix}_final.pt"

    depth_text = args.depths or args.buckets
    depth_values = torch.tensor(parse_depths(depth_text), dtype=torch.long, device=device) if depth_text else torch.unique(train["depths"].long()).to(device)
    print({"train_depth_min": int(depth_values.min().item()), "train_depth_max": int(depth_values.max().item()), "train_depth_count": int(depth_values.numel())})
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        y_epoch = epoch_depths(depth_values, args.steps_per_epoch * args.batch_size, rng)
        for start in range(0, y_epoch.numel(), args.batch_size):
            y = y_epoch[start:start + args.batch_size].float()
            x, _ = random_walks(target, moves, inverse, y.long(), rng)
            pred = model(x)
            loss = F.mse_loss(pred.float(), y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            seen += x.size(0)
        val_metrics = eval_model(model, val["states"], val["depths"], args.batch_size, device)
        row = {"epoch": epoch, "train_mse": total / max(seen, 1), "states_seen": seen, **{f"val_{k}": v for k, v in val_metrics.items()}}
        append_csv(log_path, row)
        if val_metrics["mse"] < best:
            best = val_metrics["mse"]
            save_checkpoint(best_path, model, {"kind": "state", "metric": metric, "epoch": epoch, "val": val_metrics})
        print(row)

    save_checkpoint(final_path, model, {"kind": "state", "metric": metric, "epoch": args.epochs})
    latest = ROOT / "checkpoints" / f"latest_state_{metric.lower()}.txt"
    latest.write_text(str(best_path), encoding="utf-8")
    print(best_path)


if __name__ == "__main__":
    main()
