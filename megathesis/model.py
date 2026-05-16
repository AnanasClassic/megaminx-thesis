import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingSum(nn.Module):
    def __init__(self, state_size, num_classes, out_features):
        super().__init__()
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(self.state_size * self.num_classes, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("offsets", torch.arange(self.state_size, dtype=torch.long) * self.num_classes, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.state_size * self.num_classes)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        ids = x.long() + self.offsets.unsqueeze(0)
        return F.embedding_bag(ids, self.weight, mode="sum") + self.bias


class Block(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class Net(nn.Module):
    def __init__(self, state_size, num_classes, output_dim=1, hd1=1536, hd2=512, nrd=2, dropout=0.0, z_add=0):
        super().__init__()
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.output_dim = int(output_dim)
        self.z_add = int(z_add)
        self.dtype = torch.float32
        self.input = EmbeddingSum(state_size, num_classes, hd1)
        self.bn1 = nn.BatchNorm1d(hd1)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(hd1, hd2) if hd2 > 0 else None
        self.bn2 = nn.BatchNorm1d(hd2) if hd2 > 0 else None
        self.blocks = nn.ModuleList([Block(hd2, dropout) for _ in range(nrd)]) if hd2 > 0 else nn.ModuleList()
        self.output = nn.Linear(hd2 if hd2 > 0 else hd1, self.output_dim)

    def forward(self, x):
        x = self.input(x.long() + self.z_add).to(self.dtype)
        x = self.dropout(F.relu(self.bn1(x)))
        if self.hidden is not None:
            x = self.dropout(F.relu(self.bn2(self.hidden(x))))
        for block in self.blocks:
            x = block(x)
        x = self.output(x)
        return x.squeeze(-1) if self.output_dim == 1 else x


def batch_predict(model, states, device, batch_size):
    out = []
    with torch.inference_mode():
        for start in range(0, states.size(0), batch_size):
            out.append(model(states[start:start + batch_size].to(device)))
    return torch.cat([x.reshape(x.size(0), -1) if x.ndim == 1 else x for x in out], dim=0)
