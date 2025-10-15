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
import torch.nn as nn
import torch.nn.functional as F  
import random
import numpy as np

class VanillaRNN(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        
        self.config = config
        self.device = torch.device(config['device'])
        self.env = env 

        try:
            self._set_random_seeds(config['seed'])
        except: 
            pass
        
        self.vocab_size = config['vocab_size']
        self.seq_size = config['seq_size']
        
        self.num_layers = config['num_layers']
        self.units = config['units']
        self.activation = nn.ELU()
        
        self.rnn = self._build_rnn_layers()
        self.linear_layer = nn.Linear(
            in_features=self.units, 
            out_features=self.vocab_size, 
            device=self.device, 
            dtype=torch.float64
        )
        self.to(self.device)

    def _set_random_seeds(self, seed):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) # Sets the seed for the current GPU
        torch.backends.cudnn.deterministic = True # Forces use of deterministic cuDNN algorithms
        torch.backends.cudnn.benchmark = False # Disables auto-tuner for best non-deterministic algorithm

    def _build_rnn_layers(self):

        cells = []
        for l in range(self.num_layers):
            if l == 0:
                x2h = nn.Linear(in_features=self.vocab_size, out_features=self.units, bias=False, dtype=torch.float64)
            else:
                x2h = nn.Linear(in_features=self.units, out_features=self.units, bias=False, dtype=torch.float64)
            h2h = nn.Linear(in_features=self.units, out_features=self.units, bias=True, dtype=torch.float64)
            cells.append(nn.ModuleList([x2h, h2h]))
        return nn.ModuleList(cells)
    
    def one_step(self, x_t, hidden, t, mask):
        """
        x_t: (B, vocab_size) one-hot or prob vector for token at time t
        hidden: list[num_layers][seq_size] each (B, units)
        t: time step
        mask: (B, vocab_size) additive mask for logits
        """
        last_input = x_t.to(device=self.device, dtype=torch.float64)
        for l in range(self.num_layers):
            x2h, h2h = self.rnn[l]
            h_prev = hidden[l]  # h_{t-1}
            h_new = self.activation(x2h(last_input) + h2h(h_prev))
            hidden[l] = h_new
            last_input = h_new

        logits = self.linear_layer(last_input) + mask
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, hidden
    
    def forward(self, samples):
        """
        samples: (B, seq_size, vocab_size) one-hot (or soft) targets for each time step.
        Returns: (B,) sequence log-likelihood.
        """
        B = samples.shape[0]

        x_t = torch.zeros(size=(B, self.vocab_size), device=self.device, dtype=torch.float64)

        hidden = [torch.zeros(B, self.units, device=self.device, dtype=torch.float64)
          for _ in range(self.num_layers)]

        seq_logprobs = torch.zeros(B, device=self.device, dtype=torch.float64)

        for t in range(self.seq_size):
            mask = self.env.behaviorAR(samples[:, :t, :])
            log_probs_t, hidden = self.one_step(x_t, hidden, t, mask)  # (B, vocab)
            idx_t = samples[:, t, :].argmax(dim=1, keepdim=True)          # (B,1)
            token_logprob_t = log_probs_t.gather(1, idx_t).squeeze(1)     # (B,)
            seq_logprobs = seq_logprobs + token_logprob_t

            x_t = samples[:, t, :]

        return seq_logprobs

    def _assert_no_nan_params(self):
        for name, p in self.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(f"Param {name} contains NaN/Inf")
        for name, b in self.named_buffers(recurse=True):
            if not torch.isfinite(b).all():
                raise RuntimeError(f"Buffer {name} contains NaN/Inf")
    @torch.no_grad()
    def sample(self, n_samples):
        """
        Autoregressively sample sequences and return only one-hot samples.

        Returns:
            samples_onehot: (B, T, V)
        """
        B = int(n_samples)
        T = self.seq_size
        V = self.vocab_size
        self._assert_no_nan_params()

        samples_onehot = torch.zeros(B, T, V, device=self.device, dtype=torch.float64)

        x_t = torch.zeros(B, V, device=self.device, dtype=torch.float64)
        hidden = [torch.zeros(B, self.units, device=self.device, dtype=torch.float64)
          for _ in range(self.num_layers)]
        for t in range(T):
            prefix = samples_onehot[:, :t, :]      # (B, t, V)
            mask = self.env.behaviorAR(prefix)     # (B, V)
            log_probs_t, hidden = self.one_step(x_t, hidden, t, mask)  # (B, V)
            probs_t = log_probs_t.exp()
            next_idx = torch.multinomial(probs_t, num_samples=1)  # (B, 1)
            x_t.zero_().scatter_(1, next_idx, 1.0)
            samples_onehot[:, t, :] = x_t

        return samples_onehot
