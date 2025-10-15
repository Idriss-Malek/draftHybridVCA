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

from .dilatedRNN import DilatedRNN
from .vanillaRNN import VanillaRNN
from .gnn import GINAutoregressive

__all__ = ['DilatedRNN', 'VanillaRNN', 'GINAutoregressive']
