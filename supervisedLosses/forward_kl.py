import torch
def forward_kl(p_train, log_probs):
    return torch.sum(p_train * (p_train.log() - log_probs))