import json

import torch


def parse_generator_spec(data):
    if "moves" in data and "move_names" in data:
        return data["moves"], data["move_names"]
    if "actions" in data and "names" in data:
        return data["actions"], data["names"]
    raise KeyError("generator spec must contain moves/move_names or actions/names")


def compose_perm(a, b):
    return a.index_select(0, b)


def make_ftm(utm_moves, utm_names):
    lookup = {name: i for i, name in enumerate(utm_names)}
    faces = [name for name in utm_names if not name.endswith("'")]
    moves = []
    names = []
    for face in faces:
        cur = utm_moves[lookup[face]]
        powers = [cur]
        for _ in range(3):
            cur = compose_perm(cur, utm_moves[lookup[face]])
            powers.append(cur)
        labels = [face, f"{face}2", f"{face}2'", f"{face}'"]
        moves.extend(powers)
        names.extend(labels)
    return torch.stack(moves, dim=0), names


def load_moves(path, metric, device):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    moves, names = parse_generator_spec(data)
    moves = torch.tensor(moves, dtype=torch.long)
    names = list(names)
    metric = metric.upper()
    if metric == "FTM":
        moves, names = make_ftm(moves, names)
    elif metric != "UTM":
        raise ValueError("metric must be UTM or FTM")
    return moves.to(device), names


def inverse_name(name):
    if name.endswith("2'"):
        return name[:-2] + "2"
    if name.endswith("2"):
        return name + "'"
    if name.endswith("'"):
        return name[:-1]
    return name + "'"


def inverse_indices(names):
    lookup = {name: i for i, name in enumerate(names)}
    return [lookup[inverse_name(name)] for name in names]


def apply_moves(states, moves, move_ids):
    return torch.gather(states, 1, moves.index_select(0, move_ids))


def all_children(states, moves):
    state_view = states.unsqueeze(1).expand(-1, moves.size(0), -1)
    move_view = moves.unsqueeze(0).expand(states.size(0), -1, -1)
    return torch.gather(state_view, 2, move_view)


def random_walks(target, moves, inverse, depths, generator):
    device = target.device
    depths = depths.to(device=device, dtype=torch.long)
    states = target.unsqueeze(0).expand(depths.numel(), -1).clone()
    last = torch.full((depths.numel(),), -1, dtype=torch.long, device=device)
    n_moves = moves.size(0)
    for step in range(int(depths.max().item()) if depths.numel() else 0):
        active = depths > step
        if not active.any():
            break
        nxt = torch.randint(n_moves, (depths.numel(),), generator=generator, device=device)
        invalid = active & (last >= 0) & (nxt == inverse[last.clamp_min(0)])
        while invalid.any():
            nxt[invalid] = torch.randint(n_moves, (int(invalid.sum().item()),), generator=generator, device=device)
            invalid = active & (last >= 0) & (nxt == inverse[last.clamp_min(0)])
        moved = apply_moves(states, moves, nxt)
        states = torch.where(active.unsqueeze(1), moved, states)
        last = torch.where(active, nxt, last)
    return states, last


def replay(state, moves, sequence):
    current = state.unsqueeze(0)
    for move in sequence:
        current = apply_moves(current, moves, torch.tensor([int(move)], dtype=torch.long, device=current.device))
    return current.squeeze(0)
