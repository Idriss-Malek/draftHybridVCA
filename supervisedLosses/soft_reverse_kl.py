import torch
def soft_reverse_kl(p_train, log_probs):
    reverse_kl = torch.sum(log_probs.exp() * ( log_probs - p_train.log()) - log_probs.exp() + p_train) 
    negative_entropy = torch.sum(log_probs.exp() * log_probs)
    return negative_entropy + reverse_kl