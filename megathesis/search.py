import statistics
import time

import torch

from .model import batch_predict
from .moves import apply_moves, replay


def forward_flops(model):
    hd1 = model.input.weight.size(1)
    state_size = model.state_size
    flops = state_size * hd1
    hidden = hd1
    if model.hidden is not None:
        hd2 = model.hidden.out_features
        flops += 2 * hd1 * hd2
        hidden = hd2
        flops += len(model.blocks) * 4 * hd2 * hd2
    flops += 2 * hidden * model.output_dim
    return int(flops)


class BeamSolver:
    def __init__(self, model, kind, moves, names, inverse, target, device, batch_size=65536, no_reverse=True):
        self.model = model
        self.kind = kind
        self.moves = moves
        self.names = names
        self.inverse = inverse
        self.target = target
        self.device = device
        self.batch_size = int(batch_size)
        self.no_reverse = bool(no_reverse)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        self.hash_vec = torch.randint(0, int(1e15), (target.numel(),), generator=gen, dtype=torch.int64).to(device)
        self.flops_per_input = forward_flops(model)

    def hash_states(self, states):
        return (states.long() * self.hash_vec).sum(dim=1)

    def expand(self, states, paths, last_moves, visited):
        n = states.size(0)
        n_moves = self.moves.size(0)
        parents = torch.arange(n, device=self.device).repeat_interleave(n_moves)
        move_ids = torch.arange(n_moves, device=self.device).repeat(n)
        if self.no_reverse:
            parent_last = last_moves.index_select(0, parents)
            valid = (parent_last < 0) | (move_ids != self.inverse[parent_last.clamp_min(0)])
            parents = parents[valid]
            move_ids = move_ids[valid]
        if move_ids.numel() == 0:
            return None
        children = apply_moves(states.index_select(0, parents), self.moves, move_ids)
        hashes = self.hash_states(children).detach().cpu().tolist()
        keep = []
        layer = set()
        for i, h in enumerate(hashes):
            if h in visited or h in layer:
                continue
            layer.add(h)
            keep.append(i)
        if not keep:
            return None
        keep = torch.tensor(keep, dtype=torch.long, device=self.device)
        return children.index_select(0, keep), parents.index_select(0, keep), move_ids.index_select(0, keep), [hashes[i] for i in keep.cpu().tolist()], len(hashes)

    def score(self, states, children, parents, move_ids, stats):
        if self.kind == "state":
            inputs = int(children.size(0))
            stats["model_inputs"] += inputs
            stats["model_flops"] += inputs * self.flops_per_input
            return batch_predict(self.model, children, self.device, self.batch_size).view(-1).float()
        inputs = int(states.size(0))
        stats["model_inputs"] += inputs
        stats["model_flops"] += inputs * self.flops_per_input
        q = batch_predict(self.model, states, self.device, self.batch_size).float()
        return q[parents, move_ids]

    def solve(self, initial, beam_width, max_depth):
        started = time.perf_counter()
        state = initial.to(self.device)
        if torch.equal(state, self.target):
            return self.result(True, [], started, "solved", {})
        states = state.unsqueeze(0)
        paths = [[]]
        last_moves = torch.full((1,), -1, dtype=torch.long, device=self.device)
        visited = set(self.hash_states(states).detach().cpu().tolist())
        stats = {"expanded_states": 0, "generated_candidates": 0, "unique_candidates": 0, "model_inputs": 0, "model_flops": 0}
        for _ in range(max_depth):
            expanded = self.expand(states, paths, last_moves, visited)
            stats["expanded_states"] += int(states.size(0))
            if expanded is None:
                return self.result(False, None, started, "empty", stats)
            children, parents, move_ids, hashes, generated = expanded
            stats["generated_candidates"] += int(generated)
            stats["unique_candidates"] += int(children.size(0))
            scores = self.score(states, children, parents, move_ids, stats)
            solved = (children == self.target).all(dim=1)
            if solved.any():
                i = int(torch.nonzero(solved, as_tuple=False)[0].item())
                path = paths[int(parents[i].item())] + [int(move_ids[i].item())]
                return self.result(True, path, started, "solved", stats)
            k = min(int(beam_width), int(children.size(0)))
            chosen = torch.topk(scores, k=k, largest=False, sorted=False).indices
            states = children.index_select(0, chosen)
            parent_cpu = parents.index_select(0, chosen).cpu().tolist()
            move_cpu = move_ids.index_select(0, chosen).cpu().tolist()
            hash_cpu = [hashes[int(i)] for i in chosen.cpu().tolist()]
            paths = [paths[p] + [m] for p, m in zip(parent_cpu, move_cpu)]
            last_moves = move_ids.index_select(0, chosen)
            visited.update(hash_cpu)
        return self.result(False, None, started, "max_depth", stats)

    def result(self, success, path, started, reason, stats):
        elapsed = time.perf_counter() - started
        stats = {**{"expanded_states": 0, "generated_candidates": 0, "unique_candidates": 0, "model_inputs": 0, "model_flops": 0}, **stats}
        stats["time"] = elapsed
        stats["evaluated_moves_per_sec"] = stats["generated_candidates"] / max(elapsed, 1e-9)
        stats["model_inputs_per_sec"] = stats["model_inputs"] / max(elapsed, 1e-9)
        stats["model_flops_per_sec"] = stats["model_flops"] / max(elapsed, 1e-9)
        return {"success": bool(success), "reason": reason, "moves": path, "length": None if path is None else len(path), **stats}

    def verify(self, initial, path):
        if path is None:
            return False
        return torch.equal(replay(initial.to(self.device), self.moves, path), self.target)


def summarize(results):
    success = [r for r in results if r["success"]]
    times = [float(r["time"]) for r in results]
    lengths = [int(r["length"]) for r in success]
    total_time = sum(times)
    out = {
        "tests": len(results),
        "solved": len(success),
        "success_rate": len(success) / max(len(results), 1),
        "mean_time": statistics.fmean(times) if times else 0.0,
        "median_time": statistics.median(times) if times else 0.0,
        "p90_time": sorted(times)[min(len(times) - 1, int(0.9 * len(times)))] if times else 0.0,
        "mean_length": statistics.fmean(lengths) if lengths else None,
        "median_length": statistics.median(lengths) if lengths else None,
    }
    for key in ("expanded_states", "generated_candidates", "unique_candidates", "model_inputs", "model_flops"):
        out[key] = sum(int(r[key]) for r in results)
    out["evaluated_moves_per_sec"] = out["generated_candidates"] / max(total_time, 1e-9)
    out["model_inputs_per_sec"] = out["model_inputs"] / max(total_time, 1e-9)
    out["model_flops_per_sec"] = out["model_flops"] / max(total_time, 1e-9)
    out["mean_model_flops"] = out["model_flops"] / max(len(results), 1)
    return out
