"""
inpainting.py — Energy Matching inpainting (paper Section 3.2, Algorithm 3).

Implements Algorithm 3 exactly: sweeps t from 0 → tau_s with ε(t) increasing
from 0 → epsilon_max following equation (3) of the paper. Observed pixels are
enforced via a soft measurement-fidelity energy term (not hard clamping), which
is consistent with the flow-matching trajectory used during generation.

Example:
    cd experiments/genmodel
    python inpainting.py \
        --checkpoint ../../results/genmodel_YYYYMMDD_HH/cifar10_checkpoint_phase1_final.pt \
        --mask_type center \
        --num_chains 4 \
        --inpaint_savedir results/inpainting
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from absl import app, flags, logging
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import config
config.define_flags()

from network_transformer_vit import (
    EBAttnModelWrapper,
    EBHopfieldModelWrapper,
    EBMLPModelWrapper,
    EBSimpleEncoderWrapper,
    EBViTModelWrapper,
)

FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------- #
# Inpainting-specific flags
# ---------------------------------------------------------------------------- #
flags.DEFINE_string("checkpoint", "", "Path to .pt checkpoint (required).")
flags.DEFINE_enum(
    "mask_type", "center", ["center", "bottom", "random"],
    "center=center block, bottom=bottom half, random=random pixels.",
)
flags.DEFINE_float("mask_fraction", 0.5,
                   "Fraction of the image side to mask. "
                   "For mask_type=center: side length of the square block as a fraction of H/W. "
                   "For mask_type=random: fraction of total pixels to mask.")
flags.DEFINE_integer("num_chains", 4, "Parallel Langevin chains per image.")
flags.DEFINE_float("tau_s_inpaint", 3.25,
                   "Sampling end-time τ_s. Chain runs from t=0 to t=tau_s. "
                   "Paper empirically observes FID plateaus at τ_s≈3.25 for CIFAR-10.")
flags.DEFINE_float("tau_star_inpaint", 1.0,
                   "Warm-up cutoff τ* from eq. (3). ε=0 for t<τ*, increases linearly "
                   "to epsilon_inpaint for τ*≤t≤1, then held at epsilon_inpaint for t>1.")
flags.DEFINE_float("dt_inpaint", 0.01,
                   "Langevin step size Δt. Steps = tau_s / dt (e.g. 325 at defaults).")
flags.DEFINE_float("epsilon_inpaint", 0.05,
                   "Maximum noise scale ε_max reached at t=1. Matches epsilon_max from training.")
flags.DEFINE_float("zeta_inpaint", 0.1,
                   "Measurement noise ζ controlling fidelity strength. "
                   "Smaller = stronger pull toward observed values. "
                   "The fidelity gradient is 2ε/ζ² · mask · (y - x).")
flags.DEFINE_float(
    "interaction_sigma", 0.0,
    "σ for inter-chain interaction energy strength. "
    "Encourages diverse completions in the inpainted region.",
)
flags.DEFINE_float(
    "interaction_mask_fraction", 1.0,
    "Fraction of the inpaint mask's bounding box to use as interaction region B. "
    "1.0 = full inpaint mask, 0.0 = no interaction. "
    "Shrinks the bounding box symmetrically toward its center.",
)
flags.DEFINE_string("inpaint_savedir", "results/inpainting", "Output directory.")
flags.DEFINE_integer("num_test_images", 8, "Number of CIFAR-10 test images to process.")
flags.DEFINE_string("input_image", "",
                    "Path to a custom image file (PNG/JPEG). "
                    "When set, --num_test_images is ignored and only this image is inpainted. "
                    "The image is resized to match the model's input resolution.")


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
# Mask generation
# ---------------------------------------------------------------------------- #
def make_mask(H, W, device):
    """Returns (H, W) bool tensor — True = pixel is MISSING (to inpaint)."""
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    if FLAGS.mask_type == "center":
        sz = int(H * FLAGS.mask_fraction)
        r0, c0 = (H - sz) // 2, (W - sz) // 2
        mask[r0: r0 + sz, c0: c0 + sz] = True
    elif FLAGS.mask_type == "bottom":
        mask[H // 2:, :] = True
    else:  # random
        n = int(H * W * FLAGS.mask_fraction)
        mask.view(-1)[torch.randperm(H * W, device=device)[:n]] = True
    return mask


def make_interaction_mask(inpaint_mask):
    """
    Returns a (H, W) float tensor B ⊆ inpaint_mask for the interaction energy.

    interaction_mask_fraction controls the size of B relative to the bounding
    box of inpaint_mask:
      1.0 → B = full inpaint mask  (same behaviour as before)
      0.0 → B = empty (no interaction)
      0.5 → inner 50% of the bounding box (by linear dimension), intersected
             with the inpaint mask so B never leaks into observed pixels.
    """
    frac = FLAGS.interaction_mask_fraction
    if frac <= 0.0:
        return None
    if frac >= 1.0:
        return inpaint_mask.float()

    rows = inpaint_mask.any(dim=1).nonzero(as_tuple=True)[0]
    cols = inpaint_mask.any(dim=0).nonzero(as_tuple=True)[0]
    r_min, r_max = rows[0].item(), rows[-1].item()
    c_min, c_max = cols[0].item(), cols[-1].item()

    r_center = (r_min + r_max) / 2.0
    c_center = (c_min + c_max) / 2.0
    r_half = (r_max - r_min) * frac / 2.0
    c_half = (c_max - c_min) * frac / 2.0

    H, W = inpaint_mask.shape
    B = torch.zeros(H, W, dtype=torch.float, device=inpaint_mask.device)
    r_lo = max(0, round(r_center - r_half))
    r_hi = min(H, round(r_center + r_half) + 1)
    c_lo = max(0, round(c_center - c_half))
    c_hi = min(W, round(c_center + c_half) + 1)
    B[r_lo:r_hi, c_lo:c_hi] = 1.0
    return B * inpaint_mask.float()  # ensure B ⊆ inpaint_mask


# ---------------------------------------------------------------------------- #
# ε schedule — equation (3) from the paper
# ---------------------------------------------------------------------------- #
def epsilon_at(t, eps_max, tau_star):
    """
    ε(t) from eq. (3):
      0             for t < tau_star
      linearly 0→eps_max  for tau_star ≤ t ≤ 1
      eps_max       for t > 1
    """
    if t < tau_star:
        return 0.0
    if t <= 1.0:
        return eps_max * (t - tau_star) / max(1.0 - tau_star, 1e-8)
    return eps_max


# ---------------------------------------------------------------------------- #
# Langevin inpainting (Algorithm 3 from the paper)
# ---------------------------------------------------------------------------- #
def run_inpainting(x_orig, inpaint_mask, model, device):
    """
    x_orig:       (C, H, W) in [-1, 1]
    inpaint_mask: (H, W) bool, True = pixel to inpaint

    Returns (num_chains, C, H, W) with diverse completions.

    Follows Algorithm 3 exactly:
      - t sweeps from 0 → tau_s (same as generation)
      - ε(t) increases 0 → epsilon_inpaint following eq. (3)
      - Observed pixels enforced via soft measurement fidelity:
            U(x) = V(x) + ε/ζ² · ||y - A(x)||²
        giving an extra gradient term: +2ε/ζ² · mask_obs · (y - x)
      - Optional interaction energy for diverse completions
    """
    N = FLAGS.num_chains
    dt = FLAGS.dt_inpaint
    tau_s = FLAGS.tau_s_inpaint
    tau_star = FLAGS.tau_star_inpaint
    eps_max = FLAGS.epsilon_inpaint
    zeta = FLAGS.zeta_inpaint
    sigma = FLAGS.interaction_sigma

    N_steps = max(1, int(round(tau_s / dt)))
    logging.info(
        f"Langevin: {N_steps} steps × dt={dt} = T={N_steps * dt:.2f}  "
        f"tau_star={tau_star}  ε 0→{eps_max}  ζ={zeta}"
    )

    # Observed values repeated across chains: (N, C, H, W)
    y_obs = x_orig.unsqueeze(0).expand(N, -1, -1, -1).clone().to(device)

    # All chains start from Gaussian noise (full image, including observed pixels).
    # The measurement fidelity term will pull observed pixels toward y_obs.
    x = torch.randn_like(y_obs)

    # obs_mask4: (1, 1, H, W) float — 1 at observed pixels, 0 at masked pixels
    obs_mask4 = (~inpaint_mask).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Interaction mask B: sub-region of the inpainted area where diversity is encouraged.
    B_2d = make_interaction_mask(inpaint_mask) if sigma > 0.0 else None
    B = B_2d.unsqueeze(0).unsqueeze(0) if B_2d is not None else None  # (1,1,H,W)

    snapshots = {}  # step -> (N, C, H, W) tensor

    for step in range(N_steps):
        t = step * dt
        epsilon_t = epsilon_at(t, eps_max, tau_star)
        noise_std = math.sqrt(2.0 * dt * epsilon_t) if epsilon_t > 0.0 else 0.0

        t_tensor = torch.full((N,), t, device=device)

        with torch.no_grad():
            # grad_V = ∇V(x) = -velocity(x, t)
            grad_V = -model.velocity(x, t_tensor)  # (N, C, H, W)

            # Measurement fidelity gradient: -∇_x [ε/ζ² · ||y - A(x)||²]
            #   = +2ε/ζ² · obs_mask · (y_obs - x)
            # Subtracted from grad_V because Langevin step is x -= dt * ∇U
            if epsilon_t > 0.0 and zeta > 0.0:
                fidelity_grad = -(2.0 * epsilon_t / zeta ** 2) * obs_mask4 * (y_obs - x)
                effective_grad = grad_V + fidelity_grad
            else:
                effective_grad = grad_V

            if B is not None:
                # Interaction: ∇E_int_i = -(2/σ²)·B·N·(xi - x_mean)
                x_mean = x.mean(dim=0, keepdim=True)
                interaction = (2.0 / sigma ** 2) * B * N * (x - x_mean)
                effective_grad = effective_grad - interaction

            x = x - dt * effective_grad
            if noise_std > 0.0:
                x = x + noise_std * torch.randn_like(x)
            x = x.clamp(-1.0, 1.0)

        if (step + 1) % 100 == 0:
            logging.info(f"  step {step + 1}/{N_steps}  t={t:.3f}  ε={epsilon_t:.4f}")
            snapshots[step + 1] = x.detach().cpu()

    return x, snapshots


# ---------------------------------------------------------------------------- #
# Visualisation
# ---------------------------------------------------------------------------- #
def save_result(x_orig, inpaint_mask, inpainted, savedir, idx, step=None):
    """
    Saves a single-row grid:
        [original | masked_input | chain_0 | chain_1 | ... | chain_{N-1}]
    Images are converted from [-1, 1] to [0, 1] before saving.
    """
    masked = x_orig.clone()
    masked[:, inpaint_mask] = 0.0  # gray (0 in [-1,1]) for missing pixels

    grid = torch.cat([
        x_orig.unsqueeze(0).cpu(),
        masked.unsqueeze(0).cpu(),
        inpainted.cpu(),
    ], dim=0)
    grid = (grid + 1.0) / 2.0

    os.makedirs(savedir, exist_ok=True)
    suffix = f"_step{step:04d}" if step is not None else ""
    path = os.path.join(savedir, f"inpaint_{idx:04d}{suffix}.png")
    vutils.save_image(grid, path, nrow=2 + inpainted.shape[0], padding=2, normalize=False)
    logging.info(f"Saved {path}")


def save_snapshots(x_orig, inpaint_mask, snapshots, savedir, idx):
    """Save one grid per 100-step snapshot."""
    for step, x_snap in snapshots.items():
        save_result(x_orig, inpaint_mask, x_snap, savedir, idx, step=step)


# ---------------------------------------------------------------------------- #
# Custom image loader
# ---------------------------------------------------------------------------- #
def load_image(path, device):
    """Load any PNG/JPEG, resize to the model's input size, normalize to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(img).to(device)


# ---------------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------------- #
def save_params(savedir):
    params = {
        "checkpoint": FLAGS.checkpoint,
        "input_image": FLAGS.input_image,
        "mask_type": FLAGS.mask_type,
        "mask_fraction": FLAGS.mask_fraction,
        "num_chains": FLAGS.num_chains,
        "tau_s_inpaint": FLAGS.tau_s_inpaint,
        "tau_star_inpaint": FLAGS.tau_star_inpaint,
        "dt_inpaint": FLAGS.dt_inpaint,
        "epsilon_inpaint": FLAGS.epsilon_inpaint,
        "zeta_inpaint": FLAGS.zeta_inpaint,
        "interaction_sigma": FLAGS.interaction_sigma,
        "interaction_mask_fraction": FLAGS.interaction_mask_fraction,
        "num_test_images": FLAGS.num_test_images,
        "model_type": FLAGS.model_type,
    }
    os.makedirs(savedir, exist_ok=True)
    path = os.path.join(savedir, "params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logging.info(f"Parameters saved to {path}")


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = load_model(device)

    savedir = FLAGS.inpaint_savedir
    logging.info(f"Saving results to: {os.path.abspath(savedir)}")
    save_params(savedir)

    if FLAGS.input_image:
        if not os.path.isfile(FLAGS.input_image):
            raise ValueError(f"--input_image not found: {FLAGS.input_image!r}")
        x_orig = load_image(FLAGS.input_image, device)
        _, H, W = x_orig.shape
        inpaint_mask = make_mask(H, W, device)
        logging.info(
            f"Custom image: {FLAGS.input_image}  mask_type={FLAGS.mask_type} "
            f"masked={inpaint_mask.sum().item()}/{H * W} pixels"
        )
        inpainted, snapshots = run_inpainting(x_orig, inpaint_mask, model, device)
        save_result(x_orig.cpu(), inpaint_mask.cpu(), inpainted, savedir, 0)
        save_snapshots(x_orig.cpu(), inpaint_mask.cpu(), snapshots, savedir, 0)
    else:
        dataset = CIFAR10(
            root="./data", train=False, download=True,
            transform=T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]),
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        for idx, (images, _) in enumerate(loader):
            if idx >= FLAGS.num_test_images:
                break

            x_orig = images[0].to(device)
            _, H, W = x_orig.shape
            inpaint_mask = make_mask(H, W, device)

            logging.info(
                f"[{idx + 1}/{FLAGS.num_test_images}] mask_type={FLAGS.mask_type} "
                f"masked={inpaint_mask.sum().item()}/{H * W} pixels"
            )
            inpainted, snapshots = run_inpainting(x_orig, inpaint_mask, model, device)
            save_result(x_orig.cpu(), inpaint_mask.cpu(), inpainted, savedir, idx)
            save_snapshots(x_orig.cpu(), inpaint_mask.cpu(), snapshots, savedir, idx)

    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
