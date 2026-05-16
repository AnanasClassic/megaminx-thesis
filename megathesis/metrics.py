import torch


def ranks(x):
    order = torch.argsort(x)
    out = torch.empty_like(order)
    out[order] = torch.arange(x.numel(), device=x.device)
    return out.float()


def corr(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum())
    if denom.item() == 0:
        return 0.0
    return float(((a * b).sum() / denom).item())


def regression(pred, target):
    pred = pred.float()
    target = target.float()
    diff = pred - target
    return {
        "mse": float((diff * diff).mean().item()),
        "mae": float(diff.abs().mean().item()),
        "pearson": corr(pred, target),
        "spearman": corr(ranks(pred.flatten()), ranks(target.flatten())),
    }


def ranking(pred, target, ks=(1, 3, 5)):
    best = target.argmin(dim=1)
    order = pred.argsort(dim=1)
    rank_table = torch.empty_like(order)
    rank_table.scatter_(1, order, torch.arange(pred.size(1), device=pred.device).expand_as(order))
    best_rank = rank_table.gather(1, best.view(-1, 1)).float().add(1)
    out = {"mean_rank": float(best_rank.mean().item())}
    for k in ks:
        out[f"top{k}"] = float((order[:, :k] == best.view(-1, 1)).any(dim=1).float().mean().item())
    return out


def ranking_against_move(pred, moves, ks=(1, 3, 5)):
    valid = moves >= 0
    if not valid.any():
        return {**{f"top{k}": 0.0 for k in ks}, "mean_rank": 0.0}
    pred = pred[valid]
    moves = moves[valid]
    order = pred.argsort(dim=1)
    rank_table = torch.empty_like(order)
    rank_table.scatter_(1, order, torch.arange(pred.size(1), device=pred.device).expand_as(order))
    move_rank = rank_table.gather(1, moves.view(-1, 1)).float().add(1)
    out = {"mean_rank": float(move_rank.mean().item())}
    for k in ks:
        out[f"top{k}"] = float((order[:, :k] == moves.view(-1, 1)).any(dim=1).float().mean().item())
    return out
