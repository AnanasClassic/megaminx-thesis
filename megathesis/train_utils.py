import csv
from pathlib import Path

import torch


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def batches(size, batch_size, shuffle, device, seed):
    del device
    idx = torch.arange(size)
    if shuffle:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        idx = idx[torch.randperm(size, generator=gen)]
    for start in range(0, size, batch_size):
        yield idx[start:start + batch_size]


def parse_depths(text):
    depths = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            depths.extend(range(a, b + 1))
        else:
            depths.append(int(part))
    return depths


def model_args(model):
    return {
        "state_size": model.state_size,
        "num_classes": model.num_classes,
        "output_dim": model.output_dim,
        "hd1": model.input.weight.size(1),
        "hd2": 0 if model.hidden is None else model.hidden.out_features,
        "nrd": len(model.blocks),
        "dropout": model.dropout.p,
        "z_add": model.z_add,
    }


def save_checkpoint(path, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_args": model_args(model), **meta}, path)
