import os
import random
import torch
import pandas as pd
import numpy as np
# from huggingface_hub import hf_hub_download

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1].parent))

from absl import app, flags
import config
config.define_flags()
FLAGS = flags.FLAGS

from utils_proteins import Encoder, check_duplicates, plot_epsilon
from model_proteins import Unet1DModelWrapper, VAE
from oracle import eval, BaseCNN

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(ckpt_path, scenario, task):
    """Load EBM model from a local checkpoint. Prefers ema_model key."""
    model = Unet1DModelWrapper(
        dim=28,
        channels=1,
        dim_mults=(1, 2),
        dropout=FLAGS.dropout,
        output_scale=FLAGS.output_scale,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and 'ema_model' in ckpt:
        model.load_state_dict(ckpt['ema_model'])
        print(f"Loaded ema_model from {ckpt_path}")
    elif isinstance(ckpt, dict) and 'net_model' in ckpt:
        model.load_state_dict(ckpt['net_model'])
        print(f"Loaded net_model from {ckpt_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded model from {ckpt_path}")

    model.eval()
    return model


def sample(ckpt_path, scenario, task, modality, zeta, t1,
           n_samples=512, sigma_W=None, top_k=128, n_seeds=5):
    """
    Classifier-guided posterior sampling using trained EBM.

    Args:
        ckpt_path:  Path to local .pt checkpoint file.
        scenario:   Dataset name, e.g. "aav".
        task:       Split name, "hard" or "medium".
        modality:   Predictor type, "smoothed" or "unsmoothed".
        zeta:       Classifier guidance strength (lower = stronger guidance).
        t1:         End time for SDE integration (>1.0 for extra exploration).
        n_samples:  Number of latent samples per seed.
        sigma_W:    Repulsion coefficient (None = disabled).
        top_k:      Keep top-k sequences by predictor fitness before oracle eval.
        n_seeds:    Number of random seeds to average over.
    """
    seed_all(42)
    seeds = np.random.randint(0, 1000, size=n_seeds)

    fitness_all = np.zeros(n_seeds)
    diversity_all = np.zeros(n_seeds)
    novelty_all = np.zeros(n_seeds)

    seq_len = 28
    latent_dim = 16
    y_min, y_max = 0.0, 19.5365
    y_gt = torch.ones((n_samples, 1, 1), dtype=torch.float32, device=device)

    # VAE
    encoder_path = Path(cwd) / 'vae' / f"vae_{scenario}_{task}.pt"
    vae_model = VAE(input_dim=seq_len, latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(
        torch.load(encoder_path, map_location=device)["state_dict"], strict=True
    )
    vae_model.eval()

    # EBM
    model = load_model(ckpt_path, scenario, task)

    # Predictor (for classifier guidance)
    pred = BaseCNN().to(device)
    pred_ckpt = Path(cwd) / 'predictor' / modality / f'predictor_{scenario}_{task}.ckpt'
    pred_state = torch.load(pred_ckpt, map_location=device)
    pred.load_state_dict(
        {k.replace('predictor.', ''): v for k, v in pred_state['state_dict'].items()}
    )
    pred.eval()

    dt = 0.01
    times = torch.arange(0.0, t1 + 1e-6, dt, device=device)

    for i, s in enumerate(seeds):
        np.random.seed(int(s))
        x = torch.randn(n_samples, 1, latent_dim, device=device).permute(0, 2, 1)

        for t_val in times:
            e = plot_epsilon(float(t_val))
            e_t = torch.tensor(e, device=device)
            dt_t = torch.tensor(dt, device=device)

            x.requires_grad_(True)
            v = model.potential(x, t_val.unsqueeze(0)).sum()
            y_pred = (pred.forward_soft(vae_model.decode(x.squeeze())) - y_min) / (y_max - y_min)
            likelihood = 0.5 * ((y_pred - y_gt.squeeze()) ** 2)
            cg = (e_t / zeta ** 2) * likelihood.sum()

            if sigma_W is not None:
                x_flat = x.view(n_samples, -1)
                diffs = ((x_flat.unsqueeze(1) - x_flat.unsqueeze(0)) ** 2).sum(-1)
                mask = ~torch.eye(n_samples, dtype=torch.bool, device=device)
                W = 0.5 * diffs[mask].sum() * (e_t / sigma_W ** 2)
            else:
                W = 0.0

            u = v + cg - W
            grad_x = torch.autograd.grad(u, x)[0]

            noise = torch.randn_like(x)
            sigma = torch.sqrt(2.0 * e_t * dt_t)
            with torch.no_grad():
                x = (x - dt_t * grad_x + sigma * noise).clamp(-1, 1)

        # Decode
        x_dec = vae_model.decode(x.squeeze())
        seqs = Encoder().decode(torch.argmax(x_dec, -1))
        _, seqs = check_duplicates(seqs)

        # Top-k filter by predictor
        fitness_preds = []
        with torch.no_grad():
            for seq in seqs:
                f = pred.forward(Encoder().encode(seq).to(device)[None, ...]).item()
                fitness_preds.append(f)
        fitness_tensor = torch.tensor(fitness_preds)
        k = min(top_k, len(seqs))
        _, idxs = torch.topk(fitness_tensor, k)
        seqs = [seqs[idx] for idx in idxs.tolist()]

        # Save and evaluate with oracle
        outpath = os.path.join(cwd, 'results', 'samples.csv')
        pd.Series(seqs).to_csv(outpath, index=False, header=False)

        f_med, div, nov_med = eval(
            scenario=scenario,
            task=task,
            baselines_samples_dir=outpath,
        )
        fitness_all[i] = f_med
        diversity_all[i] = div
        novelty_all[i] = nov_med
        print(f"  Seed {i+1}/{n_seeds}: fitness={f_med:.4f}, diversity={div:.4f}, novelty={nov_med:.4f}")

    return fitness_all, diversity_all, novelty_all


def main(argv):
    scenario = "aav"
    task = "hard"

    # Checkpoint to evaluate
    ckpt_path = os.path.join(
        cwd, "results", "EBMTime_20260605_04",
        "EBMTime_aav_hard_weights_step_11000.pt"
    )

    # Paper hyperparameters for AAV hard
    zeta = 0.009
    t1 = 1.3
    modality = "smoothed"

    print(f"\nEvaluating: {ckpt_path}")
    print(f"Task: {scenario} {task} | zeta={zeta} | t1={t1} | modality={modality}\n")

    fitness_all, diversity_all, novelty_all = sample(
        ckpt_path=ckpt_path,
        scenario=scenario,
        task=task,
        modality=modality,
        zeta=zeta,
        t1=t1,
    )

    print(f"\n{'='*40}")
    print(f"Fitness   {fitness_all.mean():.4f} ± {fitness_all.std():.4f}")
    print(f"Diversity {diversity_all.mean():.4f} ± {diversity_all.std():.4f}")
    print(f"Novelty   {novelty_all.mean():.4f} ± {novelty_all.std():.4f}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    app.run(main)
