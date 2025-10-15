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

import torch
from torch import Tensor

from .env import Env


class MaximumDiversity(Env):
    """
    Maximum diversity selection environment.

    Given a set of points, we select K elements that maximize pairwise diversity.
    The energy is defined as the negative sum of pairwise distances between the
    selected elements, making the problem suitable for minimization routines.
    """

    def __init__(self, points: Tensor, K: int, distance: str = "euclidean"):
        """
        Args:
            points: Tensor of shape (N, D) or (N,) representing candidate items.
            K: Number of elements to select.
            distance: Distance metric to use ('euclidean' or 'cosine').
        """
        if points.ndim == 1:
            points = points.unsqueeze(-1)

        if points.ndim != 2:
            raise ValueError("`points` must be a 2D tensor of shape (N, D) or a 1D vector.")

        if not (1 <= K <= points.shape[0]):
            raise ValueError("`K` must satisfy 1 <= K <= N.")

        self.points = points.clone().detach()
        self.K = int(K)
        self.N = points.shape[0]
        self.distance = distance.lower()

        if self.distance not in {"euclidean", "cosine"}:
            raise ValueError("Unsupported distance metric. Choose 'euclidean' or 'cosine'.")

    def neighbor(self, solution: Tensor, index1: Tensor, index2: Tensor) -> Tensor:
        """
        Swap two positions in the selection mask/permutation.
        Expects solution as one-hot selection of shape (B, N) with exactly K ones.
        """
        if solution.ndim != 2:
            raise ValueError("solution must have shape (B, N).")

        B, N = solution.shape
        if N != self.N:
            raise ValueError("solution width must match environment N.")

        idx1 = index1.to(dtype=torch.long).clamp(0, N - 1)
        idx2 = index2.to(dtype=torch.long).clamp(0, N - 1)

        if idx1.ndim == 0:
            idx1 = idx1.expand(B)
        if idx2.ndim == 0:
            idx2 = idx2.expand(B)

        rows = torch.arange(B, device=solution.device)
        new_solution = solution.clone()
        tmp = new_solution[rows, idx1].clone()
        new_solution[rows, idx1] = new_solution[rows, idx2]
        new_solution[rows, idx2] = tmp

        return new_solution

    @torch.no_grad()
    def energy(self, solution: Tensor) -> Tensor:
        """
        Compute energy as the negative sum of pairwise distances.
        Solutions should be (B, N) binary masks with exactly K ones per row.
        """
        if solution.ndim != 2:
            raise ValueError("solution must have shape (B, N).")
 
        B, N = solution.shape
        if N != self.N:
            raise ValueError("solution width must match environment N.")

        selection_counts = solution.sum(dim=1)
        expected = torch.full((B,), float(self.K), device=solution.device, dtype=solution.dtype)
        if not torch.allclose(selection_counts, expected, atol=1e-4):
            raise ValueError("Each solution must select exactly K elements.")
 
        points = self.points.to(device=solution.device, dtype=solution.dtype)

        distances = self._pairwise_distances(points)
        selected = solution.bool()

        energy = []
        for mask in selected:
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() <= 1:
                energy.append(torch.tensor(0.0, device=solution.device, dtype=solution.dtype))
                continue

            pairwise = distances[idx][:, idx]
            energy.append(-pairwise.triu(diagonal=1).sum())

        return torch.stack(energy)

    def _pairwise_distances(self, points: Tensor) -> Tensor:
        if self.distance == "euclidean":
            expanded = points.unsqueeze(0) - points.unsqueeze(1)
            return torch.linalg.norm(expanded, dim=-1)
        elif self.distance == "cosine":
            norm = torch.linalg.norm(points, dim=1, keepdim=True)
            normalized = points / norm.clamp(min=1e-12)
            return 1.0 - normalized @ normalized.t()
        raise ValueError("Unsupported distance metric.")

    def canonicalize(self, input: Tensor) -> Tensor:
        return input

    def behaviorAR(self, prefix_one_hot: Tensor) -> Tensor:
        """
        Enforce the hard constraint that at most K elements can be selected.
        Expects prefix_one_hot with shape (B, T, 2) where the last dimension
        represents the token {0, 1}. When a chain already contains K ones, the
        action corresponding to token 1 is masked out.
        """
        if prefix_one_hot.ndim != 3 or prefix_one_hot.shape[-1] != 2:
            raise ValueError("Expected prefix_one_hot of shape (B, T, 2) for binary decisions.")

        device = prefix_one_hot.device
        dtype = prefix_one_hot.dtype
        ones_count = prefix_one_hot[..., 1].sum(dim=1)

        mask = torch.zeros(prefix_one_hot.shape[0], 2, device=device, dtype=dtype)
        mask[ones_count >= float(self.K), 1] = float("-inf")
        return mask
