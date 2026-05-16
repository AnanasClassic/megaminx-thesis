import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .io import ROOT, build_model_from_checkpoint, load_group, resolve_device, run_id as make_run_id
from .metrics import ranking, regression
from .model import Net, batch_predict
from .moves import all_children, random_walks
from .train_utils import append_csv, batches, parse_depths, save_checkpoint


def eval_model(model, states, labels, batch_size, device):
    model.eval()
    pred = []
    with torch.inference_mode():
        for idx in batches(states.size(0), batch_size, False, device, 0):
            pred.append(model(states.index_select(0, idx).to(device)).float().cpu())
    pred = torch.cat(pred)
    return {**regression(pred, labels.float()), **ranking(pred, labels.float())}


def teacher_targets(teacher, states, moves, target, batch_size):
    children = all_children(states, moves)
    flat = children.reshape(-1, children.size(-1))
    pred = batch_predict(teacher, flat, states.device, batch_size).view(states.size(0), moves.size(0)).float()
    solved = (children == target.view(1, 1, -1)).all(dim=2)
    return torch.where(solved, pred.new_zeros(()), pred)


def sample_depths(depth_values, count, generator, device):
    idx = torch.randint(len(depth_values), (count,), generator=generator, device=device)
    return depth_values.index_select(0, idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default="")
    parser.add_argument("--val-path", default="")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--depths", default="")
    parser.add_argument("--buckets", default="")
    parser.add_argument("--teacher-batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hd1", type=int, default=1536)
    parser.add_argument("--hd2", type=int, default=512)
    parser.add_argument("--nrd", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="neighbour")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    state_path = Path(args.state_path) if args.state_path else ROOT / "datasets" / "utm" / "state_train.pt"
    val_path = Path(args.val_path) if args.val_path else state_path.with_name("neigh_val.pt")
    state_data = torch.load(state_path, map_location="cpu", weights_only=False)
    val = torch.load(val_path, map_location="cpu", weights_only=False)
    metric = state_data["metric"]
    moves, _, inverse, target = load_group(state_data["group_id"], state_data["target_id"], metric, device)
    teacher, teacher_payload = build_model_from_checkpoint(args.teacher, device)
    if teacher_payload.get("metric") != metric:
        raise ValueError(f"teacher metric {teacher_payload.get('metric')} does not match train metric {metric}")
    for param in teacher.parameters():
        param.requires_grad_(False)
    num_classes = int(target.max().item() - min(0, int(target.min().item())) + 1)
    z_add = -int(target.min().item()) if int(target.min().item()) < 0 else 0
    model = Net(target.numel(), num_classes, moves.size(0), args.hd1, args.hd2, args.nrd, args.dropout, z_add).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_id = make_run_id()
    prefix = f"{args.run_name}_{metric.lower()}_{run_id}"
    log_path = ROOT / "logs" / f"{prefix}.csv"
    best = math.inf
    best_path = ROOT / "checkpoints" / f"{prefix}_best.pt"
    final_path = ROOT / "checkpoints" / f"{prefix}_final.pt"

    depth_text = args.depths or args.buckets
    depth_values = torch.tensor(parse_depths(depth_text), dtype=torch.long, device=device) if depth_text else torch.unique(state_data["depths"].long()).to(device)
    print({"train_depth_min": int(depth_values.min().item()), "train_depth_max": int(depth_values.max().item()), "train_depth_count": int(depth_values.numel())})
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for _ in range(args.steps_per_epoch):
            depths = sample_depths(depth_values, args.batch_size, rng, device)
            x, _ = random_walks(target, moves, inverse, depths, rng)
            with torch.inference_mode():
                y = teacher_targets(teacher, x, moves, target, args.teacher_batch_size)
            y = y.clone()
            pred = model(x)
            loss = F.mse_loss(pred.float(), y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            seen += x.size(0)
        val_metrics = eval_model(model, val["states"], val["labels"], args.batch_size, device)
        row = {"epoch": epoch, "train_mse": total / max(seen, 1), "states_seen": seen, **{f"val_{k}": v for k, v in val_metrics.items()}}
        append_csv(log_path, row)
        if val_metrics["mse"] < best:
            best = val_metrics["mse"]
            save_checkpoint(best_path, model, {"kind": "neighbour", "metric": metric, "epoch": epoch, "val": val_metrics, "teacher": str(args.teacher)})
        print(row)

    save_checkpoint(final_path, model, {"kind": "neighbour", "metric": metric, "epoch": args.epochs, "teacher": str(args.teacher)})
    (ROOT / "checkpoints" / f"latest_neighbour_{metric.lower()}.txt").write_text(str(best_path), encoding="utf-8")
    print(best_path)


if __name__ == "__main__":
    main()
