import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, config, env):
        """
        layer_sizes: list of integers
            Example: [10, 20, 30, 5]
            - Input dim = 10
            - Hidden layers = [20, 30]
            - Output dim = 5
        """
        super().__init__()
        self.input_size = config['input_size']
        layer_sizes = [self.input_size] + config['layer_sizes'] + [self.input_size + 1]

        self.env = env
        self.device = torch.device(config['device'])

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ELU())
        layers = layers[:-1]  # remove last activation

        self.net = nn.Sequential(*layers)
        base_tensor = config.get('base_tensor', None)
        if base_tensor is None or base_tensor == 'zeros':
            self.base_tensor = torch.zeros((config['batch_size'], config['seq_size']), device=self.device, dtype=torch.long)
        if base_tensor == "random":
            self.base_tensor = torch.randint(0, 2, (config['batch_size'], config['seq_size']), device=self.device, dtype=torch.long)
        if base_tensor == "heuristic":
            file_path = config["heuristic_samples"]+f"/seed{config['seed']}.pt"
            original_heuristic_samples = torch.load(file_path, map_location=self.device)
            binary_heuristic_samples = original_heuristic_samples/2 + 0.5
            self.base_tensor = binary_heuristic_samples.to(dtype = torch.long)
        self.to(self.device)

    @property
    def dev(self):
        # Single source of truth for device (robust even if moved later)
        return next(self.parameters()).device

    def forward(self, one_hot: torch.Tensor):
        # Guarantee inputs are on the same device as the module
        one_hot = one_hot.to(self.dev)

        samples = one_hot.argmax(dim=-1)  # (B, N) on self.dev
        B, N = samples.shape

        # Build all new tensors on the module's device
        device = self.dev
        current   = self.base_tensor.clone()
        seq_logprobs = torch.zeros(B, device=device, dtype=torch.float64)
        done      = torch.zeros(B, device=device, dtype=torch.bool)
        last_flip = torch.full((B,), -1, device=device, dtype=torch.long)
        idx       = torch.arange(N, device=device)

        while not done.all():
            # Cast integer "current" to float for Linear
            logits = self.net(current.float())  # (B, N+1)
            if logits.shape[1] != N + 1:
                raise RuntimeError("one_step must return (B, N+1) logits with col N = 'finish'")

            invalid = (idx[None, :] <= last_flip[:, None])  # (B, N)
            logits[:, :N] = logits[:, :N].masked_fill(invalid, float('-inf'))

            logp = F.log_softmax(logits, dim=-1).to(torch.float64)

            rows = (~done).nonzero(as_tuple=False).squeeze(1)
            cur = current[rows]
            tgt = samples[rows]
            diff = (cur != tgt)

            allowed = idx[None, :] > last_flip[rows][:, None]
            allowed_mismatch = diff & allowed
            need_flip = allowed_mismatch.any(dim=1)

            actions = torch.full((rows.numel(),), N, device=device, dtype=torch.long)
            if need_flip.any():
                cols = allowed_mismatch[need_flip].float().argmax(dim=1)
                actions[need_flip] = cols

            seq_logprobs[rows] += logp[rows, actions]

            if need_flip.any():
                flip_rows = rows[need_flip]
                flip_cols = actions[need_flip]
                current[flip_rows, flip_cols] ^= 1
                last_flip[flip_rows] = flip_cols

            done_rows = rows[~need_flip]
            done[done_rows] = True

        return seq_logprobs

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples binary sequences using the model's policy with:
        - actions 0..N-1 = flip bit j, action N = finish
        - monotone flip constraint: indices must strictly increase

        Returns:
            one_hot: (n_samples, N, 2) tensor of {0,1} one-hot vectors.
        """
        device = self.dev
        N = int(self.input_size)
        B = n_samples

        current   = self.base_tensor.clone()
        last_flip = torch.full((B,), -1, device=device, dtype=torch.long)
        done      = torch.zeros(B, device=device, dtype=torch.bool)
        idx       = torch.arange(N, device=device)

        while not done.all():
            logits = self.net(current.float())  # (B, N+1)
            if logits.shape[1] != N + 1:
                raise RuntimeError("one_step must return (B, N+1) logits with column N = 'finish'.")

            invalid = (idx[None, :] <= last_flip[:, None])  # (B, N)
            logits[:, :N] = logits[:, :N].masked_fill(invalid, float("-inf"))

            rows = (~done).nonzero(as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                break

            act = Categorical(logits=logits[rows]).sample()  # (M,)

            flip_mask = (act < N)
            if flip_mask.any():
                flip_rows = rows[flip_mask]
                flip_cols = act[flip_mask]
                current[flip_rows, flip_cols] ^= 1
                last_flip[flip_rows] = flip_cols

            finish_rows = rows[~flip_mask]
            if finish_rows.numel() > 0:
                done[finish_rows] = True

        one_hot = F.one_hot(current.to(torch.int64), num_classes=2)  # (B, N, 2)
        return one_hot
