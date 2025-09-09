import torch
def forward_kl(p_train, log_probs):
    return torch.sum(log_probs.exp() * ( log_probs - p_train.log()) - log_probs.exp() + p_train) 