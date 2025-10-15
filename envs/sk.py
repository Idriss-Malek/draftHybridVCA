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
from .env import Env

class SherringtonKirkpatrick(Env):
    """
    SK model (zero field):
        E(s) = - s^T J s,   s_i in {-1, +1}
    One-hot representation: X in {0,1}^{B x N x 2}, channels encode {-1, +1}.
    """

    def __init__(self, J: torch.Tensor):
        """
        J: (N,N) coupling matrix. We'll symmetrize and zero the diagonal (standard SK).
        """
        J = 0.5 * (J + J.T).contiguous()
        J.fill_diagonal_(0)
        self.J = J
        self.N = J.shape[0]

    # ---------- helpers ----------
    @staticmethod
    def _onehot_to_spins(X: torch.Tensor) -> torch.Tensor:
        """
        (B,N,2) -> (B,N) in {-1,+1}, via s = X[...,1] - X[...,0].
        """
        return X[..., 1] - X[..., 0]

    @staticmethod
    def _normalize_index(idx: torch.Tensor, B: int, device) -> torch.Tensor:
        """
        Accept scalar or (B,) long; return (B,) long on `device`.
        """
        idx = idx.to(device=device, dtype=torch.long)
        if idx.dim() == 0:
            idx = idx.expand(B)
        return idx

    def neighbor(self, solution: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Flip a single spin at 'index' by swapping the two channels at that site.
        solution: (B,N,2), index: scalar or (B,)
        returns: (B,N,2)
        """
        X = solution
        B, N, _ = X.shape
        dev = X.device
        idx = self._normalize_index(index, B, dev)

        rows = torch.arange(B, device=dev)
        X_new = X.clone()
        # grab the slice (B,2) at the chosen indices
        X_new[rows, idx] = X_new[rows, idx][:, [1, 0]]
        return X_new

    @torch.no_grad()
    def energy(self, solution: torch.Tensor) -> torch.Tensor:
        """
        E = -1/2s^T J s  (batchwise).
        """
        X = solution
        s = self._onehot_to_spins(X).to(dtype=self.J.dtype)  # (B,N)
        Js = s @ self.J                                      # (B,N)
        return - 0.5 * (s * Js).sum(dim=1)                    # (B,)

    def canonicalize(self, X: torch.Tensor) -> torch.Tensor:
        """
        Z2-canonicalization by *global* flip:
          If the first spin is -1, flip the entire configuration.
          This maps X to an element in its orbit (either s or -s).
          Example: [-1, +1, -1] -> [+1, -1, +1].
        """
        rows_to_flip = (X[:, 0, 0] == 1)   # (B,)
        if not rows_to_flip.any():
            return X

        Xc = X.clone()
        flip_rows = rows_to_flip.nonzero(as_tuple=False).squeeze(1)
        Xc[flip_rows] = Xc[flip_rows][..., [1, 0]]  # swap channels
        return Xc

    def behaviorAR(self, prefix_one_hot: torch.Tensor) -> torch.Tensor:
        """
        No constraints for SK. Always return a zero mask (B,2).
        (Do not use behavioral masking to break symmetry.)
        """
        B = prefix_one_hot.shape[0]
        dev = prefix_one_hot.device
        dtype = prefix_one_hot.dtype
        return torch.zeros((B, 2), device=dev, dtype=dtype)

    