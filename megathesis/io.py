import json
import time
from pathlib import Path

import torch

from .model import Net
from .moves import inverse_indices, load_moves


ROOT = Path(__file__).resolve().parents[1]


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def torch_load(path, **kwargs):
    return torch.load(path, **kwargs)


def load_target(group_id, target_id, device):
    path = ROOT / "data" / "targets" / f"p{int(group_id):03d}-t{int(target_id):03d}.pt"
    return torch_load(path, map_location=device, weights_only=True)


def load_group(group_id, target_id, metric, device):
    moves, names = load_moves(ROOT / "data" / "generators" / f"p{int(group_id):03d}.json", metric, device)
    target = load_target(group_id, target_id, device)
    inverse = torch.tensor(inverse_indices(names), dtype=torch.long, device=device)
    return moves, names, inverse, target


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_id():
    return int(time.time())


def build_model_from_checkpoint(path, device):
    payload = torch_load(path, map_location=device, weights_only=False)
    model = Net(**payload["model_args"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def latest_checkpoint(kind, metric):
    path = ROOT / "checkpoints" / f"latest_{kind}_{metric.lower()}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path.read_text(encoding="utf-8").strip()
