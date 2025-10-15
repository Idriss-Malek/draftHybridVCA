import sys
from pathlib import Path

_repo_root = Path(__file__).resolve()
for parent in _repo_root.parents:
    if (parent / ".git").exists():
        _repo_root = parent
        break
else:
    _repo_root = _repo_root.parent

_repo_root_str = str(_repo_root)
if _repo_root_str not in sys.path:
    sys.path.append(_repo_root_str)

import sys
import os
import argparse
import json
import csv
import os
from datetime import datetime
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))

from envs import TSP
from models import DilatedRNN, VanillaRNN
from supervisedLosses import *

import torch
import torch.optim as optim
import torch.nn.functional as F


tsp_instances_dir = _repo_root / "data" / "TSP Instances"
tsp_data = {
    64: str(tsp_instances_dir / "coordinates_N64.txt"),
    128: str(tsp_instances_dir / "coordinates_N128.txt"),
    256: str(tsp_instances_dir / "coordinates_N256.txt"),
}


def main(config):
    record = {}
    coords = []
    device = torch.device(config["device"])
    with open(tsp_data[config["seq_size"]], "r") as f:
        lines = f.readlines()
        n = int(lines[0].strip())  # first line = number of coordinates
        for line in lines[1:]:
            x, y = map(float, line.strip().split())
            coords.append([x, y])

    coordinates = torch.tensor(coords, dtype=torch.float64, device=device)
    env = TSP(coordinates)
    if config["model_type"] == "DilatedRNN":
        model = DilatedRNN(config, env)
        model_opt = DilatedRNN(config, env)
    if config["model_type"] == "VanillaRNN":
        model = VanillaRNN(config, env)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    original_weights = copy.deepcopy(model)

    # if config["n_supervised"] is not None and config["n_supervised"] > 0:
    heuristic_samples_file = os.path.normpath(
        os.path.join(
            current_dir,
            "..",
            "heuristic_samples",
            config["heuristic"],
            "tsp",
            f"tsp{config['seq_size']}anneal1024.pt",
        )
    )
    heuristic_samples_idx = torch.load(heuristic_samples_file, map_location=device)
    heuristic_samples = F.one_hot(
        heuristic_samples_idx, num_classes=heuristic_samples_idx.size(1)
    ).to(device=device, dtype=torch.float64)
    heuristic_energies = env.energy(heuristic_samples)
    supervisedTemp = torch.std(heuristic_energies, correction=0) / 2
    p_train = F.softmax(-heuristic_energies / supervisedTemp, dim = -1)

    T0 = config["T0"]
    heurSol_top5 = torch.index_select(heuristic_samples, dim=0, index=torch.topk(heuristic_energies, 5, largest=False, sorted=True).indices)
    heurEn_top5 = torch.index_select(heuristic_energies, dim=0, index=torch.topk(heuristic_energies, 5, largest=False, sorted=True).indices)
    for t in range(config["n_supervised"]):
        optimizer.zero_grad()
        heuristic_log_probs = model(heuristic_samples)
        energies = env.energy(heuristic_samples)
        FreeEnergy = energies + T0 * heuristic_log_probs
        F_detached = FreeEnergy.detach()
        if config.get('supervised_loss', None) is None:
            loss = forward_kl(p_train, heuristic_log_probs)
        else:
            if config['supervised_loss'] == 'forward_kl':
                loss = forward_kl(p_train, heuristic_log_probs)
            if config['supervised_loss'] == 'reverse_kl':
                loss = reverse_kl(p_train, heuristic_log_probs)
            if config['supervised_loss'] == 'soft_reverse_kl':
                loss = soft_reverse_kl(p_train, heuristic_log_probs)

        loss.backward()

        optimizer.step()
    
    for t in range(config.get("n_supervised2", 0)):
        optimizer.zero_grad()
        heuristic_log_probs = model(heuristic_samples)
        loss = normalize_then_reverse_kl(p_train, heuristic_log_probs)
        loss.backward()
        optimizer.step()
    
    # supervised_weights = copy.deepcopy(model)


    for _ in range(config["n_warmup"]):

        optimizer.zero_grad()
        solutions = model.sample(config["batch_size"])
        log_probs = model(solutions)
        energies = env.energy(solutions)
        FreeEnergy = energies + T0 * log_probs
        F_detached = FreeEnergy.detach()
        loss = torch.mean(log_probs * F_detached) - torch.mean(log_probs) * torch.mean(
            F_detached
        )
        loss.backward()
        optimizer.step()

    # with torch.no_grad():
    #     record[0] = float('inf')
    #     for _ in range(1000):
    #         solutions = model.sample(config["batch_size"])
    #         energies = env.energy(solutions)
    #         record[0] = min(torch.min(energies).item(), record[0])

    # Annealing step
    clip_eps = config.get("ppo_clip_eps", 0.2)
    ppo_epochs = config.get("iter_per_temp", 4)       
    ent_coef   = config.get("entropy_coef", 0.1)
    target_kl  = config.get("target_kl", 0.02)  
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    n_transition = int(0.5 * config["n_anneal"])
    for t in range(config["n_anneal"]):
        T = T0 * (1 - t / config["n_anneal"])
        model_old = copy.deepcopy(model).eval()
        for p in model_old.parameters():
            p.requires_grad_(False)

        # Collect one batch from old policy
        with torch.no_grad():
            solutions = model_old.sample(config["batch_size"])
            logp_old = model_old(solutions)
            energies = env.energy(solutions)
            # advantage-like signal from your free energy with old logp
            Free_energy = energies + T * logp_old
            adv = (Free_energy.mean() - Free_energy).detach()  # lower F is better ⇒ positive advantage

        # PPO epochs over this batch
        for _ in range(ppo_epochs):
            optimizer.zero_grad()
            logp_new = model(solutions)
            ratio = torch.exp(logp_new - logp_old)     # importance sampling ratio

            # surrogate objective (maximize)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            ppo_obj = torch.min(surr1, surr2).mean()

            entropy = -(logp_new).mean()
            loss = -(ppo_obj) - ent_coef * entropy     # negate to minimize

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # optional early stop if KL blew up
            with torch.no_grad():
                approx_kl = (logp_old - logp_new).mean().abs().item()
                if approx_kl > 1.5 * target_kl:
                    break
 
    # from torch.nn.utils import parameters_to_vector, vector_to_parameters

    # def _device_of(model):
    #     return next(model.parameters()).device

    # def diag_fisher_selfsample(model_star,
    #                         n_batches: int = 100,
    #                         batch_size: int = 128,
    #                         clamp_min: float = 1e-12):
    #     """
    #     Estimates diag(F(theta*)) using x ~ p_{theta*} via model_star.sample(batch_size).
    #     Works for generative models where model(x) returns log_prob(x) (score function training).
    #     """
    #     model_star.eval()
    #     device = _device_of(model_star)

    #     diagF = None
    #     for _ in range(n_batches):
    #         # 1) Sample from p_{theta*}
    #         with torch.no_grad():
    #             x = model_star.sample(batch_size)  # shape depends on your model
    #             x = x.to(device)

    #         # 2) Compute mean negative log-likelihood and its gradient wrt theta*
    #         model_star.zero_grad(set_to_none=True)
    #         logp = model_star(x)               # expected: log p_{theta*}(x)
    #         nll = (-logp).mean()
    #         nll.backward()

    #         # 3) Collect grads and accumulate squared grads (empirical Fisher diag)
    #         g = []
    #         for p in model_star.parameters():
    #             if p.requires_grad:
    #                 g.append(torch.zeros_like(p) if p.grad is None else p.grad.detach())
    #         gvec = parameters_to_vector(g)
    #         diag = gvec * gvec
    #         diagF = diag if diagF is None else (diagF + diag)

    #     diagF = diagF / max(1, n_batches)
    #     return diagF.clamp_min(clamp_min)

    # def fisher_dist2(model, model_star, diagF):
    #     v = parameters_to_vector([p.detach() for p in model.parameters()]) - \
    #         parameters_to_vector([p.detach() for p in model_star.parameters()])
    #     return torch.sum(v*v*diagF).item()
    # def fisher_cosine(model1, model2, model_star, diagF, eps: float = 1e-12):
   
    #     vec1 = parameters_to_vector([p.detach() for p in model1.parameters()])
    #     vec2 = parameters_to_vector([p.detach() for p in model2.parameters()])
    #     vec_star = parameters_to_vector([p.detach() for p in model_star.parameters()])

    #     v1 = vec1 - vec_star
    #     v2 = vec2 - vec_star

    #     # Ensure diagF on same device/dtype
    #     diagF = diagF.to(v1.device).type_as(v1)

    #     # Fisher inner products
    #     num = torch.sum(v1 * v2 * diagF)

    #     n1_sq = torch.sum(v1 * v1 * diagF)
    #     n2_sq = torch.sum(v2 * v2 * diagF)

    #     denom = torch.sqrt(n1_sq.clamp_min(eps)) * torch.sqrt(n2_sq.clamp_min(eps))
    #     if denom.abs() < eps:
    #         return 0.0  # both vectors effectively zero under the Fisher metric

    #     # Clamp for numerical safety into [-1, 1]
    #     cosine = (num / denom).item()
    #     if cosine > 1.0:
    #         cosine = 1.0
    #     elif cosine < -1.0:
    #         cosine = -1.0
    #     return cosine
    # diagF = diag_fisher_selfsample(model)
    # print("Original weights:", fisher_dist2(original_weights, model_opt, diagF))
    # print("After Forward KL supervised warm up:", fisher_dist2(model, model_opt, diagF))
    # try:
    #     print("Cosine", fisher_cosine(original_weights,model,model_opt,diagF))
    # except:
    #     print("Cosine non available because there is a 0 norm")
    

    record["mean"] = 0.0
    record["min"] = float("inf")
    with torch.no_grad():
        for st in range(1000):
            solutions = model.sample(config["batch_size"])
            energies = env.energy(solutions)
            record["min"] = min(record["min"], torch.min(energies).item())
            record["mean"] = (record["mean"] * st + torch.mean(energies).item()) / (st + 1)

    return record


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path. If omitted, a path is generated under ./results/.",
    )
    args = parser.parse_args()

    # Load config JSON -> dict
    with open(args.config, "r") as f:
        config = json.load(f)

    out_path = args.out or os.path.join(
        current_dir,
        "results", "tsp",
        f"tsp{config['seq_size']}_{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_anneal", "min", "mean"])

    # Run
    for n_anneal in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        config["n_anneal"] = n_anneal
        result = main(config)
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n_anneal, result['min'], result['mean']])

        print(f"Saved results to {out_path}")
