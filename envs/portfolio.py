from .env import Env
import torch

class Portfolio(Env):

    def __init__(self,
                 mu: torch.Tensor,       # (N,)
                 Sigma: torch.Tensor,    # (N,N)
                 lam: float | torch.Tensor = 1.0):
        assert mu.ndim == 1, "mu must be (N,)"
        assert Sigma.ndim == 2 and Sigma.shape[0] == Sigma.shape[1] == mu.shape[0], "Sigma must be (N,N)"
        Sigma = 0.5 * (Sigma + Sigma.T).contiguous()  # symmetrize

        self.N = mu.shape[0]
        self.mu = mu
        self.Sigma = Sigma
        self.lam = float(lam) if not torch.is_tensor(lam) else lam.item()

    @classmethod
    def from_random(cls, N: int, seed: int | None = 1):
        if seed is not None:
            torch.manual_seed(seed)
        lam = torch.rand(()).item()
        A = torch.randn(max(1, N//2), N, dtype=torch.float64)
        Sigma = (A.T @ A) + 0.1*torch.eye(N, dtype=torch.float64)  
        x_star = (torch.rand(N)>0.5).to(torch.float64)
        mu = 2*lam*(Sigma @ x_star)
        E_star = lam*(x_star @ (Sigma @ x_star)) - mu @ x_star
        return cls(mu, Sigma, lam), E_star

    # ---------- helpers ----------
    @staticmethod
    def _onehot_to_bits(X: torch.Tensor) -> torch.Tensor:
        """
        (B,N,2) -> (B,N) in {0,1}, take the '1' channel.
        Also accept (B,N) and pass through.
        """
        if X.ndim == 3 and X.shape[-1] == 2:
            return X[..., 1]
        if X.ndim == 2:
            return X
        raise ValueError("Expected (B,N,2) one-hot or (B,N) bits.")

    @staticmethod
    def _norm_idx(t: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(t):
            raise TypeError("Index tensors must be torch.Tensor")
        t = t.to(device=device, dtype=torch.long)
        if t.dim() == 0:
            return t.expand(B)
        if t.shape == (B,):
            return t
        raise ValueError("Index tensors must be shape () or (B,)")

    def neighbor(self, solution: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        B, N, _ = solution.shape
        idx = index.to(solution.device, dtype=torch.long)
        b = torch.arange(B, device=solution.device)
        new = solution.clone()
        new[b, idx] = new[b, idx].roll(1, dims=-1)  # swap [0,1] <-> [1,0]
        return new

    @torch.no_grad()
    def energy(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Returns E(x) = λ x^T Σ x - μ^T x for each batch item.
        Accepts (B,N,2) one-hot or (B,N) bits.
        """
        x = self._onehot_to_bits(solution).to(dtype=torch.float64)  # (B,N)
        device = x.device

        mu = self.mu.to(device=device, dtype=torch.float64)         # (N,)
        Sigma = self.Sigma.to(device=device, dtype=torch.float64)   # (N,N)
        lam = torch.as_tensor(self.lam, device=device, dtype=torch.float64)

        # quadratic term: (x Σ x) per batch
        quad = torch.einsum('bi,ij,bj->b', x, Sigma, x)             # (B,)
        lin  = torch.einsum('bi,i->b', x, mu)                       # (B,)
        return lam * quad - lin                                     # (B,)

    def canonicalize(self, t):
        return t

    def behaviorAR(self, partial_solution: torch.Tensor) -> torch.Tensor:
        B = partial_solution.shape[0]
        return torch.ones(B, 2, dtype=torch.bool, device=partial_solution.device)
