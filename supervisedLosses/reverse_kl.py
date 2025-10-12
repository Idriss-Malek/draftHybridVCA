import torch
import torch.nn.functional as F
def reverse_kl(p_train, log_probs):
    probs = log_probs.exp()
    return torch.sum(probs * ( log_probs - p_train.log()) - probs + p_train)