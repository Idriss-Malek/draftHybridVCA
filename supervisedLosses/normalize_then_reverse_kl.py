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
import torch.nn.functional as F
def normalize_then_reverse_kl(p_train, log_probs):
    p_uniform = torch.ones_like(p_train)/p_train.shape[-1]
    l1Norm = torch.sum(log_probs.exp(), dim=-1)
    normalized = (1-l1Norm)*p_uniform + l1Norm*p_train
    return torch.sum(normalized * ( normalized.log() - p_train.log())) 