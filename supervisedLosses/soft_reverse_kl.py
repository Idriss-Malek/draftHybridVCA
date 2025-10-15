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
def soft_reverse_kl(p_train, log_probs):
    reverse_kl = torch.sum(log_probs.exp() * ( log_probs - p_train.log()) - log_probs.exp() + p_train) 
    negative_entropy = torch.sum(log_probs.exp() * log_probs)
    return negative_entropy + reverse_kl