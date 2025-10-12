import torch
def naive(p_train, log_probs):
    return - (p_train * log_probs).mean()