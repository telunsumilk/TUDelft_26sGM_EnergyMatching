###############################################################################
# File: generate_mnist_dataset.py
#
# Purpose:
#   Generate a reusable dataset of individual MNIST samples from a pretrained
#   Energy Matching model. This mirrors experiments/cifar10/generate_cifar_dataset.py;
#   only the checkpoint architecture defaults are changed for the MNIST model.
#
# Example:
#   python3 experiments/mnist/generate_mnist_dataset.py \
#       --resume_ckpt=results-mnist/EBMTime_20260605_01/checkpoint_50400.pt \
#       --num_samples=10 \
#       --batch_size=16 \
#       --t_end=1.0 \
#       --dt_gibbs=0.01 \
#       --use_ema=True \
#       --epsilon_max=0.01 \
#       --time_cutoff=1.0 \
#       --output_dir=./sampling_results
###############################################################################
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torchsde
from absl import app, flags, logging
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import config_multigpu as config

config.define_flags()
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_samples", 100, "Number of generated images to save.")
flags.DEFINE_float("t_end", 3.25, "Final SDE time; t_start is fixed to 0.")
flags.DEFINE_bool("use_ema", True, "If True, load EMA weights; else raw weights.")
flags.DEFINE_integer(
    "progress_chunk_steps",
    10,
    "SDE steps per tqdm progress update. Set <=0 to use one-shot sdeint per batch.",
)
flags.DEFINE_integer("seed", 42, "Random seed for generation.")
flags.DEFINE_integer("preview_count", 64, "Number of generated images in preview grid.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CIFAR_EXPERIMENT_DIR = os.path.join(REPO_ROOT, "experiments", "cifar10")
sys.path.insert(0, CIFAR_EXPERIMENT_DIR)

from network_transformer_vit import EBViTModelWrapper

sys.path.append(REPO_ROOT)
from utils_cifar_imagenet import plot_epsilon


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def solve_sde_heun(model, x, t_start, t_end, dt=0.01, progress_chunk_steps=10):
    """Integrate x from t_start to t_end with Stratonovich Euler-Heun."""
    if t_end <= t_start:
        return x

    orig_shape, batch_size = x.shape, x.size(0)
    x_flat = x.view(batch_size, -1)

    class _FlattenSDE(torchsde.SDEStratonovich):
        def __init__(self, net):
            super().__init__(noise_type="diagonal")
            self.net = net

        def f(self, t, y):
            y_unflat = y.view(*orig_shape)
            return self.net(t.expand(batch_size).to(y.device), y_unflat).view(batch_size, -1)

        def g(self, t, y):
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            e_tensor = torch.tensor(e_val, device=y.device, dtype=y.dtype)
            return torch.sqrt(2.0 * e_tensor).expand_as(y)

    sde = _FlattenSDE(model)
    ts = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)

    with torch.no_grad():
        if progress_chunk_steps <= 0:
            x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
            x_flat = x_sol[-1]
        else:
            total_steps = len(ts) - 1
            for start in tqdm(
                range(0, total_steps, progress_chunk_steps),
                desc="Sampling SDE",
                unit="chunk",
                leave=False,
            ):
                end = min(start + progress_chunk_steps, total_steps)
                x_sol = torchsde.sdeint(
                    sde,
                    x_flat,
                    ts[start : end + 1],
                    method="heun",
                    dt=dt,
                )
                x_flat = x_sol[-1]

    return x_flat.view(*orig_shape).clamp(-1, 1)


def build_model(device):
    ch_mult = config.parse_channel_mult(FLAGS)
    model = EBViTModelWrapper(
        dim=(3, 32, 32),
        num_channels=FLAGS.num_channels,
        num_res_blocks=FLAGS.num_res_blocks,
        channel_mult=ch_mult,
        attention_resolutions=FLAGS.attention_resolutions,
        num_heads=FLAGS.num_heads,
        num_head_channels=FLAGS.num_head_channels,
        dropout=FLAGS.dropout,
        output_scale=FLAGS.output_scale,
        energy_clamp=FLAGS.energy_clamp,
        patch_size=4,
        embed_dim=FLAGS.embed_dim,
        transformer_nheads=FLAGS.transformer_nheads,
        transformer_nlayers=FLAGS.transformer_nlayers,
    ).to(device)
    return model.eval()


def load_checkpoint(model, device):
    if not FLAGS.resume_ckpt or not os.path.isfile(FLAGS.resume_ckpt):
        raise FileNotFoundError("--resume_ckpt is missing or invalid.")

    ckpt = torch.load(FLAGS.resume_ckpt, map_location=device)
    key = "ema_model" if FLAGS.use_ema else "net_model"
    model.load_state_dict(ckpt[key], strict=True)
    logging.info(f"Loaded {key} from {FLAGS.resume_ckpt}")


def write_metadata(run_dir, device):
    metadata = {
        "script": "experiments/mnist/generate_mnist_dataset.py",
        "resume_ckpt": FLAGS.resume_ckpt,
        "checkpoint_weights": "ema_model" if FLAGS.use_ema else "net_model",
        "num_samples": FLAGS.num_samples,
        "batch_size": FLAGS.batch_size,
        "t_end": FLAGS.t_end,
        "dt_gibbs": FLAGS.dt_gibbs,
        "time_cutoff": FLAGS.time_cutoff,
        "epsilon_max": FLAGS.epsilon_max,
        "progress_chunk_steps": FLAGS.progress_chunk_steps,
        "seed": FLAGS.seed,
        "device": str(device),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main(_):
    seed_all(FLAGS.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = datetime.now().strftime("mnist_em_%Y%m%d_%H%M%S")
    run_dir = os.path.join(FLAGS.output_dir, run_name)
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    logging.info(f"Generated images will be saved in: {images_dir}")
    logging.info(f"Using device: {device}")

    model = build_model(device)
    load_checkpoint(model, device)
    write_metadata(run_dir, device)

    saved = 0
    preview_images = []
    n_batches = math.ceil(FLAGS.num_samples / FLAGS.batch_size)

    for _ in tqdm(range(n_batches), desc="Generating batches", unit="batch"):
        curr_bsz = min(FLAGS.batch_size, FLAGS.num_samples - saved)
        if curr_bsz <= 0:
            break

        x = torch.randn(curr_bsz, 3, 32, 32, device=device)
        x = solve_sde_heun(
            model,
            x,
            0.0,
            FLAGS.t_end,
            dt=FLAGS.dt_gibbs,
            progress_chunk_steps=FLAGS.progress_chunk_steps,
        )
        x_01 = ((x + 1.0) / 2.0).clamp(0, 1).detach().cpu()

        for i, img in enumerate(x_01):
            image_index = saved + i
            save_image(img, os.path.join(images_dir, f"{image_index:06d}.png"))

        if len(preview_images) < FLAGS.preview_count:
            remaining = FLAGS.preview_count - len(preview_images)
            preview_images.extend([img for img in x_01[:remaining]])

        saved += curr_bsz

    if preview_images:
        preview = torch.stack(preview_images)
        nrow = max(1, int(math.sqrt(len(preview_images))))
        grid = make_grid(preview, nrow=nrow, padding=2)
        save_image(grid, os.path.join(run_dir, "grid_preview.png"))

    logging.info(f"Saved {saved} images to {images_dir}")
    logging.info(f"Preview grid saved to {os.path.join(run_dir, 'grid_preview.png')}")


if __name__ == "__main__":
    app.run(main)
