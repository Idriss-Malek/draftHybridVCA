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
def reverse_kl(p_train, log_probs):
    probs = log_probs.exp()
    return torch.sum(probs * ( log_probs - p_train.log()) - probs + p_train)