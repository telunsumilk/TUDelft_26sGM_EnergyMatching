"""
generate.py — Standalone image generation from a pre-trained Energy Matching checkpoint.

Uses the Stratonovich Heun SDE integrator (same as FID evaluation) to generate
images from Gaussian noise.

Training flags (model architecture, epsilon_max, time_cutoff, …) are automatically
read from the train.INFO log file in the checkpoint directory, so you only need to
pass --checkpoint. Any flag you supply explicitly overrides the log.

Example:
    cd experiments/genmodel
    python generate.py \\
        --checkpoint ../../results/genmodel_YYYYMMDD_HH/cifar10_checkpoint_phase1_final.pt \\
        --n_samples 64 \\
        --t_end 3.25 \\
        --savedir results/generated
"""

import os
import re
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
from fid import (
    solve_sde_heun,
    simulate_image_trajectory,
    simulate_image_trajectory_dense,
    plot_image_trajectories,
    plot_pca_trajectories,
)

FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------- #
# Generation-specific flags
# ---------------------------------------------------------------------------- #
flags.DEFINE_string("checkpoint", "", "Path to .pt checkpoint (required).")
flags.DEFINE_integer("n_samples", 64, "Number of images to generate.")
flags.DEFINE_float("t_end", 3.25, "SDE integration end time.")
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
flags.DEFINE_bool("save_pca", True,
                  "Save a PCA trajectory plot (noise → generated in 2D projection).")
flags.DEFINE_integer("pca_samples", 256, "Number of samples for the PCA trajectory.")
flags.DEFINE_integer("pca_n_highlight", 15, "Number of individual paths to highlight in red.")
flags.DEFINE_integer("pca_record_every", 10, "Record a PCA frame every N SDE steps.")


# ---------------------------------------------------------------------------- #
# Auto-load flags from train.INFO
# ---------------------------------------------------------------------------- #

# Flags we care about recovering from the log (architecture + SDE schedule).
# Explicit command-line overrides always win.
_RECOVER_FLAGS = [
    "model_type",
    "num_channels", "num_res_blocks", "channel_mult",
    "attention_resolutions", "num_heads", "num_head_channels",
    "dropout", "output_scale", "energy_clamp",
    "embed_dim", "transformer_nheads", "transformer_nlayers",
    "hopfield_memories", "hopfield_beta",
    "epsilon_max", "time_cutoff",
    "dataset",
]

# absl log line format:  "I0608 12:34:56.789 12345 train.py:445]   key = value"
_LOG_LINE_RE = re.compile(r'\]\s+(\w+) = (.+)$')


def _parse_log(log_path):
    """Return dict of flag_name -> raw_string_value from a train.INFO file."""
    result = {}
    with open(log_path) as f:
        for line in f:
            m = _LOG_LINE_RE.search(line.rstrip())
            if m:
                result[m.group(1)] = m.group(2)
    return result


def apply_flags_from_log(checkpoint_path, argv_flags):
    """
    Look for train.INFO next to the checkpoint and apply recovered flags.
    Flags already supplied on the command line (present in argv_flags) are
    left untouched so explicit overrides always win.
    """
    log_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "train.INFO")
    if not os.path.isfile(log_path):
        logging.warning(f"train.INFO not found at {log_path} — using default flags.")
        return

    logging.info(f"Reading training flags from {log_path}")
    parsed = _parse_log(log_path)
    applied = []
    for name in _RECOVER_FLAGS:
        if name in argv_flags:
            continue  # explicit override, skip
        if name not in parsed:
            continue
        try:
            FLAGS[name].parse(parsed[name])
            applied.append(f"{name}={parsed[name]}")
        except Exception as e:
            logging.warning(f"Could not apply {name}={parsed[name]!r} from log: {e}")

    if applied:
        logging.info(f"Recovered from train.INFO: {', '.join(applied)}")


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

    # Recover training flags from the log before building the model.
    # argv contains the raw command-line arguments; extract flag names from it
    # so we know which ones were explicitly set by the user.
    argv_flag_names = {a.lstrip("-").split("=")[0] for a in sys.argv[1:] if a.startswith("-")}
    if FLAGS.checkpoint:
        apply_flags_from_log(FLAGS.checkpoint, argv_flag_names)

    logging.info(f"Device: {device}")
    logging.info(f"model_type={FLAGS.model_type}  epsilon_max={FLAGS.epsilon_max}  "
                 f"time_cutoff={FLAGS.time_cutoff}  t_end={FLAGS.t_end}")

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

    if FLAGS.save_pca:
        x_pca = torch.randn(FLAGS.pca_samples, 3, 32, 32, device=device)
        frames_pca = simulate_image_trajectory_dense(
            model, x_pca, FLAGS.t_end, dt=FLAGS.dt,
            record_every=FLAGS.pca_record_every,
        )
        pca_path = os.path.join(FLAGS.savedir, "pca_trajectory.png")
        plot_pca_trajectories(frames_pca, n_highlight=FLAGS.pca_n_highlight, savepath=pca_path)
        logging.info(f"PCA trajectory saved to {pca_path}")

        stride = max(1, len(frames_pca) // 10)
        frames_pca_grid = [(t, imgs[:FLAGS.pca_n_highlight]) for t, imgs in frames_pca[::stride]]
        pca_grid_path = os.path.join(FLAGS.savedir, "pca_trajectory_grid.png")
        plot_image_trajectories(frames_pca_grid, n_samples=FLAGS.pca_n_highlight,
                                savepath=pca_grid_path)
        logging.info(f"PCA companion grid saved to {pca_grid_path}")

    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
