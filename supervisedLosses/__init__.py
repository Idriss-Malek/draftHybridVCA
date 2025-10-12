from .forward_kl import forward_kl
from .reverse_kl import reverse_kl
from .naive import naive
from .soft_reverse_kl import soft_reverse_kl
from .normalize_then_reverse_kl import normalize_then_reverse_kl

__all__ = ['forward_kl', 'reverse_kl', 'soft_reverse_kl', 'normalize_then_reverse_kl']