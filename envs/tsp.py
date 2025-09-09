from .env import Env
import torch

class TSP(Env):
    def __init__(self, coordinates: torch.Tensor):
        self.N = coordinates.shape[0]
        self.coordinates = coordinates

    def neighbor(self,
                 solution: torch.Tensor,
                 index1: torch.Tensor,
                 index2: torch.Tensor) -> torch.Tensor:
        """
        Apply a classic 2-opt move to permutation matrix/matrices (column-block reversal).

        Notes:
            - If i == j, the operation is a no-op.
        """

        B, n, _ = solution.shape
        device = solution.device

        def norm_idx(t: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(t):
                raise TypeError("`index1`/`index2` must be torch.Tensor")
            t = t.to(device=device, dtype=torch.long)
            if t.dim() == 0:
                return t.expand(B)
            if t.shape == (B,):
                return t
            raise ValueError("Index tensors must be shape () or (B,)")

        i = norm_idx(index1)
        j = norm_idx(index2)

        s = torch.minimum(i, j) + 1
        e = torch.maximum(i, j)

        s = s.clamp(0, n - 1)
        e = e.clamp(0, n - 1)

        # Build per-batch **row** remapping
        pos = torch.arange(n, device=device).unsqueeze(0).expand(B, n)  # (B, n)
        su = s.unsqueeze(1)  # (B, 1)
        eu = e.unsqueeze(1)  # (B, 1)

        in_block = (pos >= su) & (pos <= eu)
        rev_map = torch.where(in_block, eu - (pos - su), pos).long()  # (B, n)

        # Gather **rows** according to rev_map
        row_index = rev_map.unsqueeze(2).expand(B, n, n)  # (B, n, n)
        return torch.gather(solution, dim=1, index=row_index)
    
    @torch.no_grad()
    def energy(self, solution: torch.Tensor) -> torch.Tensor:
        coords = self.coordinates.to(solution.device, dtype=solution.dtype)
        T = torch.matmul(solution, coords)
        diffs = T.roll(shifts=-1, dims=1) - T  
        lengths = torch.linalg.norm(diffs, dim=2).sum(dim=1)
        return lengths

    def canonicalize(self, t):
        """
        Rotate each row so the first occurrence of token 0 moves to column 0.
        Keeps tours rotation-invariant (same as your original code).
        """
        zero_indices = torch.argmax((t == 0).int(), dim=1)                   # [B]
        col_indices = torch.arange(self.N, device=t.device)                  # [N]
        shifts = zero_indices.unsqueeze(1)                                   # [B,1]
        rolled_indices = (col_indices + shifts) % self.N                     # [B,N]
        result = torch.gather(t, 1, rolled_indices)
        return result
    
    def behaviorAR(self, prefix_one_hot: torch.Tensor) -> torch.Tensor:
        """
        prefix_one_hot: (B, T, V) one-hot (or prob) for visited tokens so far
        returns: (B, V) additive mask
        """
        visited = prefix_one_hot.any(dim=1)                   # (B, V) bool
        mask = torch.zeros_like(visited, dtype=torch.float64) # (B, V)
        mask[visited] = float('-inf')
        return mask

    def neighbor2(self,
              solution: torch.Tensor,
              index1: torch.Tensor,
              index2: torch.Tensor) -> torch.Tensor:
        """
        Swap two cities i and j in a permutation matrix/matrices (i.e., swap columns).
        If i == j, this is a no-op.

        Args:
            solution: (B, n, n) permutation matrix (or batch of them)
            index1: scalar tensor or (B,) long tensor
            index2: scalar tensor or (B,) long tensor
        Returns:
            (B, n, n) tensor with the two columns swapped per batch item
        """
        B, n, _ = solution.shape
        device = solution.device

        def norm_idx(t: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(t):
                raise TypeError("`index1`/`index2` must be torch.Tensor")
            t = t.to(device=device, dtype=torch.long)
            if t.dim() == 0:
                return t.expand(B)
            if t.shape == (B,):
                return t
            raise ValueError("Index tensors must be shape () or (B,)")

        i = norm_idx(index1).clamp(0, n - 1)
        j = norm_idx(index2).clamp(0, n - 1)

        # Build per-batch column remapping that swaps i and j
        pos = torch.arange(n, device=device).unsqueeze(0).expand(B, n)  # (B, n)
        iu = i.unsqueeze(1)  # (B, 1)
        ju = j.unsqueeze(1)  # (B, 1)

        mask_i = (pos == iu)
        mask_j = (pos == ju)

        col_map = pos.clone()
        col_map = torch.where(mask_i, ju, col_map)  # positions at i -> j
        col_map = torch.where(mask_j, iu, col_map)  # positions at j -> i

        # Gather columns according to col_map
        col_index = col_map.unsqueeze(1).expand(B, n, n)  # (B, n, n)
        P_new = torch.gather(solution, dim=2, index=col_index)

        return P_new