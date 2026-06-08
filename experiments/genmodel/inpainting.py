"""
inpainting.py — Energy Matching inpainting demo (paper Section 3.2, Algorithm 3).

Runs Langevin sampling conditioned on observed pixels (hard constraint) with
optional interaction energy between chains for diverse completions.

Example:
    cd experiments/genmodel
    python inpainting.py \
        --checkpoint ../../results/genmodel_YYYYMMDD_HH/cifar10_checkpoint_phase1_final.pt \
        --mask_type center \
        --num_chains 4 \
        --n_inpaint_steps 300 \
        --inpaint_savedir results/inpainting
"""

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
                   "Fraction of pixels to mask (only used with mask_type=random).")
flags.DEFINE_integer("num_chains", 4, "Parallel Langevin chains per image.")
flags.DEFINE_integer("n_inpaint_steps", 300, "Langevin steps per image.")
flags.DEFINE_float("dt_inpaint", 0.005, "Langevin step size.")
flags.DEFINE_float("epsilon_inpaint", 0.01,
                   "Noise scale ε; per-step noise std = sqrt(2·dt·ε).")
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
        mask[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = True
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
# Langevin inpainting (Algorithm 3 from the paper)
# ---------------------------------------------------------------------------- #
def run_inpainting(x_orig, inpaint_mask, model, device):
    """
    x_orig: (C, H, W) in [-1, 1]
    inpaint_mask: (H, W) bool, True = pixel to inpaint

    Returns (num_chains, C, H, W) with diverse completions.

    Each Langevin step:
        grad = ∇V(x) [+ interaction term if B is not None]
        x = x - dt·grad + sqrt(2·dt·ε)·N(0,I)
        x[observed] = y_obs[observed]   # hard constraint
    Interaction energy (paper eq.): W(xi,xj) = -||B(xi-xj)||²/σ²
    where B is a sub-region of the inpainted area (see make_interaction_mask).
    Smaller sigma = stronger interaction; sigma must be > 0 when B is enabled.
    """
    N = FLAGS.num_chains
    dt = FLAGS.dt_inpaint
    noise_std = math.sqrt(2.0 * dt * FLAGS.epsilon_inpaint)
    sigma = FLAGS.interaction_sigma

    # Repeat observed image across all chains: (N, C, H, W)
    y_obs = x_orig.unsqueeze(0).expand(N, -1, -1, -1).clone().to(device)

    # Initialise: observed pixels from y_obs, masked region from N(0,1)
    mask4 = inpaint_mask.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    x = torch.where(mask4, torch.randn_like(y_obs), y_obs)

    obs4 = (~inpaint_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) True = observed
    t_dummy = torch.zeros(N, device=device)

    # Interaction mask B: sub-region of the inpainted area where diversity is encouraged.
    # interaction_mask_fraction=1.0 → full mask; 0.0 → disabled; intermediate → inner crop.
    B_2d = make_interaction_mask(inpaint_mask) if sigma > 0.0 else None
    B = B_2d.unsqueeze(0).unsqueeze(0) if B_2d is not None else None  # (1,1,H,W)

    for step in range(FLAGS.n_inpaint_steps):
        with torch.no_grad():
            # model.velocity = -∇V; uses enable_grad internally so no_grad is safe
            grad_V = -model.velocity(x, t_dummy)  # (N, C, H, W)

            if B is not None:
                # Interaction gradient: ∇E_int_i = -(2/σ²)·B·Σ_{j≠i}(xi-xj)
                #                               = -(2/σ²)·B·N·(xi - x_mean)
                # B is the interaction sub-region (⊆ inpaint mask); only pixels
                # inside B are pushed apart across chains → diverse completions.
                # NOTE: follows the paper's raw formulation. Scaling can be tricky —
                # interaction magnitude depends on image scale, number of chains, and
                # gradient magnitude, so sigma needs empirical tuning.
                x_mean = x.mean(dim=0, keepdim=True)
                interaction = (2.0 / sigma ** 2) * B * N * (x - x_mean)
                effective_grad = grad_V - interaction
            else:
                effective_grad = grad_V

            x = x - dt * effective_grad + noise_std * torch.randn_like(x)
            x = torch.where(obs4, y_obs, x)  # hard constraint on observed pixels
            x = x.clamp(-1.0, 1.0)

        if (step + 1) % 100 == 0:
            logging.info(f"  step {step + 1}/{FLAGS.n_inpaint_steps}")

    return x


# ---------------------------------------------------------------------------- #
# Visualisation
# ---------------------------------------------------------------------------- #
def save_result(x_orig, inpaint_mask, inpainted, savedir, idx):
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
    path = os.path.join(savedir, f"inpaint_{idx:04d}.png")
    vutils.save_image(grid, path, nrow=2 + inpainted.shape[0], padding=2, normalize=False)
    logging.info(f"Saved {path}")


# ---------------------------------------------------------------------------- #
# Custom image loader
# ---------------------------------------------------------------------------- #
def load_image(path, device):
    """Load any PNG/JPEG, resize to the model's input size, normalize to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    # model input is 32×32 for CIFAR-10 checkpoints; adjust dim in build_model for other sizes
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(img).to(device)


# ---------------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------------- #
def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = load_model(device)

    savedir = FLAGS.inpaint_savedir
    logging.info(f"Saving results to: {os.path.abspath(savedir)}")

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
        inpainted = run_inpainting(x_orig, inpaint_mask, model, device)
        save_result(x_orig.cpu(), inpaint_mask.cpu(), inpainted, savedir, 0)
    else:
        dataset = CIFAR10(
            root="./data", train=False, download=True,
            transform=T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]),
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

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
            inpainted = run_inpainting(x_orig, inpaint_mask, model, device)
            save_result(x_orig.cpu(), inpaint_mask.cpu(), inpainted, savedir, idx)

    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
