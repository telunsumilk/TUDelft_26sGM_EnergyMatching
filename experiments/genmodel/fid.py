"""
fid.py — FID evaluation for Energy Matching models.

Importable:
    from fid import compute_fid
    results = compute_fid(ema_model, FLAGS, device, savedir)

Standalone:
    python fid.py --dataset=cifar10 --resume_ckpt=path/to/checkpoint.pt \\
        --epsilon_max=0.01 --fid_num_gen=50000
"""

import os
import sys

import torch
import torchsde
from absl import app, flags, logging
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils_cifar_imagenet import create_timestamped_dir, plot_epsilon


# --------------------------------------------------------------------------- #
# Heun SDE solver
# --------------------------------------------------------------------------- #

def solve_sde_heun(model, x, t_start, t_end, dt=0.01):
    """Stratonovich Euler-Heun integrator from t_start to t_end."""
    if t_end <= t_start:
        return x

    orig_shape = x.shape
    B = x.size(0)
    x_flat = x.view(B, -1)

    class _SDE(torchsde.SDEStratonovich):
        def __init__(self, net):
            super().__init__(noise_type="diagonal")
            self.net = net

        def f(self, t, y):
            y_img = y.view(*orig_shape)
            t_batch = t.expand(B).to(y.device)
            return self.net(t_batch, y_img).view(B, -1)

        def g(self, t, y):
            e_val = plot_epsilon(float(t))
            if e_val <= 0:
                return torch.zeros_like(y)
            scale = torch.sqrt(torch.tensor(2.0 * e_val, device=y.device, dtype=y.dtype))
            return scale.expand_as(y)

    sde = _SDE(model)
    ts = torch.arange(t_start, t_end + 1e-9, dt, device=x.device)

    with torch.no_grad():
        x_sol = torchsde.sdeint(sde, x_flat, ts, method="heun", dt=dt)
        x_final = x_sol[-1].view(*orig_shape).clamp(-1.0, 1.0)

    return x_final


# --------------------------------------------------------------------------- #
# Dataset helpers (for real-image FID updates)
# --------------------------------------------------------------------------- #

def _real_loader(flags):
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    if flags.dataset == "cifar10":
        from dataset_cifar import get_cifar10_eval_dataset
        dataset = get_cifar10_eval_dataset()
    elif flags.dataset == "imagenet32":
        from dataset_imagenet32 import ImageNet32Dataset
        dataset = ImageNet32Dataset(split="train", transform=T.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {flags.dataset!r}")

    return DataLoader(dataset, batch_size=flags.batch_size,
                      num_workers=flags.num_workers, shuffle=False, drop_last=False)


# --------------------------------------------------------------------------- #
# Trajectory visualisation  (image analog of toy2d plot_trajectories_custom)
# --------------------------------------------------------------------------- #

def simulate_image_trajectory(model, x_init, times, dt=0.01):
    """
    Run the Heun SDE on x_init, capturing the image state at each time in `times`.

    Returns a list of (time, images) tuples starting with (0.0, x_init),
    mirroring simulate_piecewise_length from the toy2d notebook.
    """
    frames = [(0.0, x_init.detach().cpu())]
    x = x_init
    t_prev = 0.0
    for t_end in times:
        x = solve_sde_heun(model, x, t_prev, t_end, dt=dt)
        frames.append((t_end, x.detach().cpu()))
        t_prev = t_end
    return frames


def plot_pca_trajectories(frames, n_highlight=10, savepath=None):
    """
    PCA projection of the SDE trajectory — direct image analog of
    plot_trajectories_custom from the toy2d notebook.

    All images are flattened to vectors, PCA reduces to 2D, then we
    scatter-plot with the same colour scheme:
      - black squares  : initial noise    (t=0)
      - olive dots     : intermediate steps
      - blue stars     : final images
      - red lines      : n_highlight individual paths
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    n_times = len(frames)
    B = frames[0][1].shape[0]

    # Flatten every frame to (B, D) and stack → (n_times * B, D)
    flat = [imgs.view(B, -1).numpy() for _, imgs in frames]
    all_points = np.concatenate(flat, axis=0)

    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_points)          # (n_times * B, 2)
    traj_2d = all_2d.reshape(n_times, B, 2)         # (n_times, B, 2)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Intermediate trajectory points (olive) — same as toy2d
    if n_times > 2:
        mid = traj_2d[1:-1]                         # (n_times-2, B, 2)
        ax.scatter(mid[:, :, 0].ravel(), mid[:, :, 1].ravel(),
                   s=0.5, alpha=0.15, c="olive", rasterized=True)

    # Initial positions — black squares
    ax.scatter(traj_2d[0, :, 0], traj_2d[0, :, 1],
               s=6, alpha=0.8, c="black", marker="s")

    # Final positions — blue stars
    ax.scatter(traj_2d[-1, :, 0], traj_2d[-1, :, 1],
               s=8, alpha=1.0, c="royalblue", marker="*")

    # Highlighted individual paths — red lines
    for i in range(min(n_highlight, B)):
        ax.plot(traj_2d[:, i, 0], traj_2d[:, i, 1],
                c="red", linewidth=1.2, alpha=0.9)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("SDE trajectory — PCA projection", fontsize=10)

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_image_trajectories(frames, n_samples=8, savepath=None):
    """
    Image analog of plot_trajectories_custom from the toy2d notebook.

    Layout:  rows = samples,  columns = time steps (t=0 … t=T)
      - First column (t=0, pure noise): black border  ← toy2d black squares
      - Middle columns (intermediate):  no border     ← toy2d olive trajectory
      - Last column (final image):      blue border   ← toy2d blue stars

    Args:
        frames:    list of (float, Tensor[B,3,32,32]) from simulate_image_trajectory
        n_samples: how many sample rows to show (≤ batch size)
        savepath:  if given, save the figure as a PNG; otherwise call plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    n_times = len(frames)
    fig, axes = plt.subplots(
        n_samples, n_times,
        figsize=(n_times * 1.4, n_samples * 1.4),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for col, (t, imgs) in enumerate(frames):
        is_first = col == 0
        is_last  = col == n_times - 1
        for row in range(n_samples):
            ax = axes[row, col]
            img = imgs[row]                        # (3, 32, 32) in [-1, 1]
            img = ((img + 1.0) / 2.0).clamp(0, 1) # → [0, 1]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            # coloured border matching toy2d colour scheme
            color = "black" if is_first else ("royalblue" if is_last else None)
            if color:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if row == 0:
                ax.set_title(f"t={t:.2f}", fontsize=7, pad=2)

    # legend matching toy2d colours
    legend = [
        mpatches.Patch(color="black",     label="noise (t=0)"),
        mpatches.Patch(color="olive",     label="intermediate"),
        mpatches.Patch(color="royalblue", label="generated"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------- #
# Main FID computation
# --------------------------------------------------------------------------- #

def compute_fid(model, flags, device, savedir):
    """
    Sweep over flags.fid_times using a sequential Heun SDE and compute FID
    at each time point.

    Real images are loaded once and fed to all FID meters.
    Fake images are generated in batches; the SDE state is carried forward
    from one time point to the next within each batch.

    Returns dict mapping float time -> FID score.
    """
    model.eval()
    times = [float(t) for t in flags.fid_times]
    fid_meters = {t: FrechetInceptionDistance(feature=2048).to(device) for t in times}

    # -- real images --------------------------------------------------------
    logging.info("FID: updating with real images...")
    for imgs, _ in tqdm(_real_loader(flags), desc="real"):
        imgs = imgs.to(device)
        # imgs from eval dataset are in [0, 1] (ToTensor only, no normalize)
        imgs_u8 = (imgs * 255).clamp(0, 255).to(torch.uint8)
        for t in times:
            fid_meters[t].update(imgs_u8, real=True)

    # -- fake images --------------------------------------------------------
    logging.info(f"FID: generating {flags.fid_num_gen} samples across {len(times)} times...")
    n_generated = 0
    with tqdm(total=flags.fid_num_gen, desc="fake") as pbar:
        while n_generated < flags.fid_num_gen:
            bsz = min(flags.batch_size, flags.fid_num_gen - n_generated)
            x = torch.randn(bsz, 3, 32, 32, device=device)
            t_prev = 0.0
            for t_end in times:
                x = solve_sde_heun(model, x, t_prev, t_end, dt=flags.dt_gibbs)
                x_01 = (x + 1.0) / 2.0
                x_u8 = (x_01 * 255).clamp(0, 255).to(torch.uint8)
                fid_meters[t_end].update(x_u8, real=False)
                t_prev = t_end
            n_generated += bsz
            pbar.update(bsz)

    # -- compute & log ------------------------------------------------------
    results = {}
    for t in times:
        score = fid_meters[t].compute().item()
        results[t] = score
        logging.info(f"  FID @ t={t:.3f} = {score:.4f}")

    # save summary
    summary_path = os.path.join(savedir, "fid_results.txt")
    with open(summary_path, "w") as f:
        for t, score in results.items():
            f.write(f"{t:.4f}\t{score:.6f}\n")
    logging.info(f"FID results written to {summary_path}")

    # -- trajectory visualisations (toy2d analogs) -------------------------
    logging.info("FID: generating trajectory visualisations...")

    # Image grid: 8 samples (rows) × time steps (columns)
    x_grid = torch.randn(8, 3, 32, 32, device=device)
    frames_grid = simulate_image_trajectory(model, x_grid, times, dt=flags.dt_gibbs)
    grid_path = os.path.join(savedir, "sde_trajectory_grid.png")
    plot_image_trajectories(frames_grid, n_samples=8, savepath=grid_path)
    logging.info(f"Image grid saved to {grid_path}")

    # PCA scatter: 256 samples give PCA enough points for meaningful structure
    x_pca = torch.randn(256, 3, 32, 32, device=device)
    frames_pca = simulate_image_trajectory(model, x_pca, times, dt=flags.dt_gibbs)
    pca_path = os.path.join(savedir, "sde_trajectory_pca.png")
    plot_pca_trajectories(frames_pca, n_highlight=10, savepath=pca_path)
    logging.info(f"PCA trajectory plot saved to {pca_path}")

    model.train()
    return results


# --------------------------------------------------------------------------- #
# Standalone entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import config
    config.define_flags()
    flags.DEFINE_bool("use_ema", True, "Load EMA weights from checkpoint.")

    def _main(argv):
        FLAGS = flags.FLAGS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        savedir = create_timestamped_dir(FLAGS.output_dir, "fid")
        logging.get_absl_handler().use_absl_log_file(program_name="fid", log_dir=savedir)
        logging.set_verbosity(logging.INFO)
        logging.info(f"Output directory: {savedir}")
        logging.info(f"Device: {device}")

        if not FLAGS.resume_ckpt or not os.path.exists(FLAGS.resume_ckpt):
            raise ValueError(f"--resume_ckpt not found: {FLAGS.resume_ckpt!r}")

        from network_transformer_vit import EBViTModelWrapper
        from config import parse_channel_mult
        ch_mult = parse_channel_mult(FLAGS)

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

        ckpt = torch.load(FLAGS.resume_ckpt, map_location=device)
        key = "ema_model" if FLAGS.use_ema else "net_model"
        model.load_state_dict(ckpt[key], strict=True)
        model.eval()
        logging.info(f"Loaded {'EMA' if FLAGS.use_ema else 'net'} weights from {FLAGS.resume_ckpt}")

        compute_fid(model, FLAGS, device, savedir)

    app.run(_main)
