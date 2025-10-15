"""
Graph neural network policies compatible with the Env interface.

This module provides an autoregressive policy that mirrors the API of the
existing RNN-based models (forward returns log-likelihood, sample generates
one-hot sequences) while replacing the recurrent backbone with a GIN graph
network. The graph structure is derived from `env.N` (assumed to be the
sequence length / vocabulary size) and a full sweep over all nodes is used
when evaluating logits.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve()
for parent in _repo_root.parents:
    if (parent / ".gitignore").exists():
        _repo_root = parent
        break
else:
    _repo_root = _repo_root.parent

_repo_root_str = str(_repo_root)
if _repo_root_str not in sys.path:
    sys.path.append(_repo_root_str)

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GINConv, global_add_pool
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "torch-geometric is required for the GIN models. "
        "Install it via `pip install torch-geometric` (plus the matching torch-scatter "
        " / torch-sparse wheels)."
    ) from exc

from envs.env import Env


class StateEncoder(nn.Module):
    """Embeds discrete node states (e.g., {unvisited, visited, last})."""

    def __init__(self, num_embeddings: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_dim)

    def forward(self, state: Tensor) -> Tensor:
        return self.embedding(state)


class GINBackbone(nn.Module):
    """Stack of GINConv layers operating on the graph defined by the environment."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            layers.append(GINConv(mlp))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data: Data, features: Tensor) -> Tensor:
        x = features
        for conv in self.layers:
            x = conv(x, data.edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        return x


class PolicyNetwork(nn.Module):
    """Graph neural network producing logits over node actions."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.0,
        num_state_embeddings: int = 3,
    ):
        super().__init__()
        self.state_encoder = StateEncoder(num_embeddings=num_state_embeddings, hidden_dim=hidden_dim)
        self.backbone = GINBackbone(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data, state: Tensor) -> Tensor:
        encoded_state = self.state_encoder(state)
        latent = self.backbone(data, encoded_state)
        logits = self.output_head(latent).squeeze(-1)
        return logits


class FlowNetwork(nn.Module):
    """Graph neural network producing pooled flow estimates (optional)."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.0,
        num_state_embeddings: int = 3,
    ):
        super().__init__()
        self.state_encoder = StateEncoder(num_embeddings=num_state_embeddings, hidden_dim=hidden_dim)
        self.backbone = GINBackbone(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data, state: Tensor) -> Tensor:
        encoded_state = self.state_encoder(state)
        latent = self.backbone(data, encoded_state)
        if not hasattr(data, "batch"):
            raise ValueError("`data.batch` is required for pooling multiple graphs.")
        pooled = global_add_pool(latent, data.batch)
        return self.output_head(pooled).squeeze(-1)


class GINAutoregressive(nn.Module):
    """
    Autoregressive policy that mirrors the Dilated/Vanilla RNN interface while
    leveraging a GIN backbone. It consumes Env masks via `behaviorAR`, returns
    log-likelihoods under `forward`, and samples batched one-hot sequences via
    `sample`.
    """

    def __init__(self, config: Dict[str, Any], env: Env):
        super().__init__()

        self.config = config
        self.env = env
        self.device = torch.device(config["device"])

        self.seq_size = int(config["seq_size"])
        self.vocab_size = int(config["vocab_size"])
        if self.seq_size != self.vocab_size:
            raise ValueError("GINAutoregressive assumes `seq_size == vocab_size` for Env compatibility.")
        if not hasattr(env, "N") or env.N != self.vocab_size:
            raise ValueError("Environment must expose `N == vocab_size` for GINAutoregressive.")

        hidden_dim = int(config.get("hidden_dim", config.get("units", 128)))
        num_layers = int(config.get("gin_layers", config.get("num_layers", 5)))
        dropout = float(config.get("dropout", 0.0))

        self.policy = PolicyNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_state_embeddings=3,
        ).to(self.device, dtype=torch.float64)

        self.edge_index = self._complete_graph_edge_index(self.vocab_size, self.device)
        self._graph_cache: Dict[int, Data] = {}

    @staticmethod
    def _complete_graph_edge_index(num_nodes: int, device: torch.device) -> Tensor:
        idx = torch.arange(num_nodes, device=device)
        row = idx.repeat_interleave(num_nodes)
        col = idx.repeat(num_nodes)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        return edge_index.long()

    def _get_graph(self, batch_size: int) -> Data:
        data = self._graph_cache.get(batch_size)
        if data is None:
            edge = self.edge_index
            num_edges = edge.size(1)
            row = edge[0].unsqueeze(0).repeat(batch_size, 1)
            col = edge[1].unsqueeze(0).repeat(batch_size, 1)
            offsets = (torch.arange(batch_size, device=self.device) * self.vocab_size).unsqueeze(1)
            batch_edge_index = torch.stack(
                [(row + offsets).reshape(-1), (col + offsets).reshape(-1)],
                dim=0,
            )
            batch_vec = torch.arange(batch_size, device=self.device).unsqueeze(1).repeat(1, self.vocab_size).reshape(-1)
            data = Data(edge_index=batch_edge_index, batch=batch_vec, num_nodes=batch_size * self.vocab_size)
            self._graph_cache[batch_size] = data
        return data

    def _build_state_labels(self, visited: Tensor, last_indices: Tensor) -> Tensor:
        """
        Encode per-node categorical states:
            0 -> available, 1 -> already visited, 2 -> last selected node.
        """
        labels = torch.zeros_like(visited, dtype=torch.long, device=self.device)
        labels = labels.masked_fill_(visited, 1)
        valid_last = last_indices >= 0
        if valid_last.any():
            batch_ids = torch.arange(labels.size(0), device=self.device)[valid_last]
            labels[batch_ids, last_indices[valid_last]] = 2
        return labels

    def _compute_policy_logits(self, visited: Tensor, last_indices: Tensor) -> Tensor:
        batch_size = visited.size(0)
        state_labels = self._build_state_labels(visited, last_indices)
        data = self._get_graph(batch_size)
        logits = self.policy(data, state_labels.reshape(-1))
        return logits.view(batch_size, self.vocab_size)

    def forward(self, samples: Tensor) -> Tensor:
        """
        Compute log-likelihood of the provided one-hot sequences.

        Args:
            samples: (B, T, V) one-hot tensor, matching Env expectations.
        Returns:
            (B,) tensor with the sequence log-likelihood.
        """
        samples = samples.to(device=self.device, dtype=torch.float64)
        B, T, V = samples.shape
        if T != self.seq_size or V != self.vocab_size:
            raise ValueError("Input samples must have shape (batch, seq_size, vocab_size).")

        seq_logprobs = torch.zeros(B, device=self.device, dtype=torch.float64)
        visited = torch.zeros(B, V, device=self.device, dtype=torch.bool)
        last_indices = torch.full((B,), -1, device=self.device, dtype=torch.long)

        for t in range(self.seq_size):
            prefix = samples[:, :t, :]
            mask = self.env.behaviorAR(prefix).to(device=self.device, dtype=torch.float64)
            logits = self._compute_policy_logits(visited, last_indices) + mask
            log_probs_t = F.log_softmax(logits, dim=-1)

            idx_t = samples[:, t, :].argmax(dim=1, keepdim=True)
            token_logprob = log_probs_t.gather(1, idx_t).squeeze(1)
            seq_logprobs = seq_logprobs + token_logprob

            visited.scatter_(1, idx_t, True)
            last_indices = idx_t.squeeze(1)

        return seq_logprobs

    def _assert_no_nan_params(self) -> None:
        for name, param in self.named_parameters():
            if param.requires_grad and not torch.isfinite(param).all():
                raise RuntimeError(f"Parameter {name} contains NaN/Inf values.")

    @torch.no_grad()
    def sample(self, n_samples: int) -> Tensor:
        """
        Autoregressively sample one-hot sequences.

        Returns:
            (B, T, V) tensor with one-hot encoded samples.
        """
        B = int(n_samples)
        T, V = self.seq_size, self.vocab_size
        self._assert_no_nan_params()

        samples = torch.zeros(B, T, V, device=self.device, dtype=torch.float64)
        visited = torch.zeros(B, V, device=self.device, dtype=torch.bool)
        last_indices = torch.full((B,), -1, device=self.device, dtype=torch.long)

        for t in range(T):
            prefix = samples[:, :t, :]
            mask = self.env.behaviorAR(prefix).to(device=self.device, dtype=torch.float64)
            logits = self._compute_policy_logits(visited, last_indices) + mask
            probs = torch.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)
            samples[:, t, :].zero_().scatter_(1, next_idx, 1.0)

            visited.scatter_(1, next_idx, True)
            last_indices = next_idx.squeeze(1)

        return samples


__all__ = [
    "StateEncoder",
    "GINBackbone",
    "PolicyNetwork",
    "FlowNetwork",
    "GINAutoregressive",
]

