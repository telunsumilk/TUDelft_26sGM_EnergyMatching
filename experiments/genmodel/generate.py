"""
generate.py — Standalone image generation from a pre-trained Energy Matching checkpoint.

Uses the Stratonovich Heun SDE integrator (same as FID evaluation) to generate
images from Gaussian noise.

Example:
    cd experiments/genmodel
    python generate.py \\
        --checkpoint ../../results/genmodel_YYYYMMDD_HH/cifar10_checkpoint_phase1_final.pt \\
        --n_samples 64 \\
        --t_end 3.0 \\
        --savedir results/generated
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torchvision.utils as vutils
from absl import app, flags, logging

import config
config.define_flags()

from network_transformer_vit import (
    EBAttnModelWrapper,
    EBHopfieldModelWrapper,
    EBMLPModelWrapper,
    EBSimpleEncoderWrapper,
    EBViTModelWrapper,
)
from fid import solve_sde_heun, simulate_image_trajectory, plot_image_trajectories

FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------- #
# Generation-specific flags
# ---------------------------------------------------------------------------- #
flags.DEFINE_string("checkpoint", "", "Path to .pt checkpoint (required).")
flags.DEFINE_integer("n_samples", 64, "Number of images to generate.")
flags.DEFINE_float("t_end", 3.0, "SDE integration end time.")
flags.DEFINE_float("dt", 0.01, "SDE step size.")
flags.DEFINE_string("savedir", "results/generated", "Output directory.")
flags.DEFINE_bool("save_grid", True, "Save a single image grid (grid.png).")
flags.DEFINE_bool("save_individual", False, "Save each image as a separate PNG.")
flags.DEFINE_bool("save_trajectory", True,
                  "Save a trajectory grid (rows=samples, cols=time steps).")
flags.DEFINE_integer("trajectory_samples", 8,
                     "Number of sample rows to show in the trajectory grid.")
flags.DEFINE_integer("trajectory_steps", 10,
                     "Number of time columns in the trajectory grid.")


# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
def build_model(device):
    ch_mult = config.parse_channel_mult(FLAGS)
    common = dict(
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
    )
    if FLAGS.model_type == "vit":
        return EBViTModelWrapper(
            **common,
            patch_size=4,
            embed_dim=FLAGS.embed_dim,
            transformer_nheads=FLAGS.transformer_nheads,
            transformer_nlayers=FLAGS.transformer_nlayers,
        ).to(device)
    if FLAGS.model_type == "attn":
        return EBAttnModelWrapper(
            **common,
            patch_size=4,
            embed_dim=FLAGS.embed_dim,
            attn_nheads=FLAGS.transformer_nheads,
        ).to(device)
    if FLAGS.model_type == "hopfield":
        return EBHopfieldModelWrapper(
            **common,
            n_memories=FLAGS.hopfield_memories,
            embed_dim=FLAGS.embed_dim,
            hopfield_beta=FLAGS.hopfield_beta,
        ).to(device)
    if FLAGS.model_type == "cnn":
        return EBSimpleEncoderWrapper(
            dim=common["dim"],
            output_scale=common["output_scale"],
            energy_clamp=common["energy_clamp"],
        ).to(device)
    return EBMLPModelWrapper(**common).to(device)


def load_model(device):
    if not FLAGS.checkpoint or not os.path.isfile(FLAGS.checkpoint):
        raise ValueError(f"--checkpoint not found: {FLAGS.checkpoint!r}")
    model = build_model(device)
    ckpt = torch.load(FLAGS.checkpoint, map_location=device)
    state = ckpt.get("ema_model") or ckpt.get("net_model")
    if state is None:
        raise ValueError("Checkpoint must contain 'ema_model' or 'net_model'.")
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Loaded: {FLAGS.checkpoint}  step={ckpt.get('step', '?')}")
    return model


# ---------------------------------------------------------------------------- #
# Generation
# ---------------------------------------------------------------------------- #
def generate(model, device):
    """Generate FLAGS.n_samples images from noise using the Heun SDE."""
    x = torch.randn(FLAGS.n_samples, 3, 32, 32, device=device)
    x = solve_sde_heun(model, x, t_start=0.0, t_end=FLAGS.t_end, dt=FLAGS.dt)
    return x  # (N, 3, 32, 32) in [-1, 1]


# ---------------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------------- #
def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = load_model(device)
    os.makedirs(FLAGS.savedir, exist_ok=True)

    logging.info(f"Generating {FLAGS.n_samples} samples  t_end={FLAGS.t_end}  dt={FLAGS.dt}")
    images = generate(model, device)

    if FLAGS.save_grid:
        grid_path = os.path.join(FLAGS.savedir, "grid.png")
        nrow = min(8, FLAGS.n_samples)
        imgs_01 = (images + 1.0) / 2.0
        vutils.save_image(imgs_01, grid_path, nrow=nrow, normalize=False)
        logging.info(f"Grid saved to {grid_path}")

    if FLAGS.save_individual:
        imgs_01 = (images + 1.0) / 2.0
        for i, img in enumerate(imgs_01):
            p = os.path.join(FLAGS.savedir, f"sample_{i:04d}.png")
            vutils.save_image(img, p)
        logging.info(f"Saved {FLAGS.n_samples} individual images to {FLAGS.savedir}")

    if FLAGS.save_trajectory:
        n = min(FLAGS.trajectory_samples, FLAGS.n_samples)
        x_traj = torch.randn(n, 3, 32, 32, device=device)
        times = [
            FLAGS.t_end * (i + 1) / FLAGS.trajectory_steps
            for i in range(FLAGS.trajectory_steps)
        ]
        frames = simulate_image_trajectory(model, x_traj, times, dt=FLAGS.dt)
        traj_path = os.path.join(FLAGS.savedir, "trajectory.png")
        plot_image_trajectories(frames, n_samples=n, savepath=traj_path)
        logging.info(f"Trajectory grid saved to {traj_path}")

    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
