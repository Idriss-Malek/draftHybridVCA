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

from .forward_kl import forward_kl
from .reverse_kl import reverse_kl
from .naive import naive
from .soft_reverse_kl import soft_reverse_kl
from .normalize_then_reverse_kl import normalize_then_reverse_kl

__all__ = ['forward_kl', 'reverse_kl', 'soft_reverse_kl', 'normalize_then_reverse_kl']