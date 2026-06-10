"""
train.py — Unified single-GPU Energy Matching trainer.

Runs Phase 1 (OT flow matching) → Phase 2 (OT + Contrastive Divergence) →
FID evaluation in one invocation.

CIFAR-10 (defaults):
    python train.py --dataset=cifar10 --phase1_steps=145000 --phase2_steps=2000 \\
        --n_gibbs=200 --lambda_cd=1e-3 --epsilon_max=0.01

ImageNet32 (key overrides):
    python train.py --dataset=imagenet32 --lr=6e-4 --energy_clamp=10000 \\
        --use_flow_weight=False --phase1_steps=640000 --phase2_steps=2000 \\
        --n_gibbs=200 --lambda_cd=1e-3 --epsilon_max=0.01 \\
        --fid_times=0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0

Resume from phase1_final checkpoint (skip Phase 1):
    python train.py --skip_phase1 --resume_ckpt=path/to/phase1_final.pt \\
        --lambda_cd=1e-3 --n_gibbs=200 --epsilon_max=0.01
"""

import copy
import math
import os
import sys
import time

import torch
from absl import app, flags, logging
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

import config
config.define_flags()
FLAGS = flags.FLAGS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils_cifar_imagenet import (
    create_timestamped_dir,
    ema,
    flow_weight,
    generate_samples,
    gibbs_sampling_time_sweep,
    infiniteloop,
    make_lr_lambda,
    save_pos_neg_grids,
)
from network_transformer_vit import (
    EBViTModelWrapper, EBAttnModelWrapper, EBMLPModelWrapper,
    EBHopfieldModelWrapper,
)


# =========================================================================== #
# Dataset
# =========================================================================== #

def get_dataset():
    import torchvision.transforms as T
    if FLAGS.dataset == "cifar10":
        from dataset_cifar import get_cifar10_dataset
        class_indices = [int(c) for c in FLAGS.cifar_classes] or None
        return get_cifar10_dataset(class_indices=class_indices, color_jitter=FLAGS.color_jitter)
    elif FLAGS.dataset == "imagenet32":
        from dataset_imagenet32 import ImageNet32Dataset
        class_indices = [int(c) for c in FLAGS.imagenet_classes] or None
        extra = []
        if FLAGS.color_jitter > 0.0:
            extra.append(T.ColorJitter(brightness=FLAGS.color_jitter, contrast=FLAGS.color_jitter,
                                       saturation=FLAGS.color_jitter, hue=FLAGS.color_jitter / 4))
        return ImageNet32Dataset(
            split="train",
            transform=T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(), *extra,
                                  T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            class_indices=class_indices,
        )
    raise ValueError(f"Unknown dataset: {FLAGS.dataset!r}")


# =========================================================================== #
# Model
# =========================================================================== #

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
        model = EBViTModelWrapper(
            **common,
            patch_size=4,
            embed_dim=FLAGS.embed_dim,
            transformer_nheads=FLAGS.transformer_nheads,
            transformer_nlayers=FLAGS.transformer_nlayers,
        )
    elif FLAGS.model_type == "attn":
        model = EBAttnModelWrapper(
            **common,
            patch_size=4,
            embed_dim=FLAGS.embed_dim,
            attn_nheads=FLAGS.transformer_nheads,
        )
    elif FLAGS.model_type == "hopfield":
        model = EBHopfieldModelWrapper(
            **common,
            n_memories=FLAGS.hopfield_memories,
            embed_dim=FLAGS.embed_dim,
            hopfield_beta=FLAGS.hopfield_beta,
        )
    else:  # mlp
        model = EBMLPModelWrapper(**common)
    model = model.to(device)
    ema_model = copy.deepcopy(model)
    return model, ema_model


def build_optimizer(model, total_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lr_lambda(total_steps))
    return optimizer, scheduler


# =========================================================================== #
# Checkpoint helpers
# =========================================================================== #

def _ckpt_path(savedir, tag):
    return os.path.join(savedir, f"{FLAGS.dataset}_checkpoint_{tag}.pt")


def save_checkpoint(model, ema_model, optimizer, scheduler, step, savedir, tag):
    ckpt = {
        "net_model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "step": step,
    }
    path = _ckpt_path(savedir, tag)
    torch.save(ckpt, path)
    logging.info(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(model, ema_model, optimizer, scheduler, device):
    if not FLAGS.resume_ckpt or not os.path.exists(FLAGS.resume_ckpt):
        return 0
    logging.info(f"Resuming from: {FLAGS.resume_ckpt}")
    ckpt = torch.load(FLAGS.resume_ckpt, map_location=device)
    model.load_state_dict(ckpt["net_model"])
    ema_model.load_state_dict(ckpt["ema_model"])
    optimizer.load_state_dict(ckpt["optim"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    scheduler.load_state_dict(ckpt["sched"])
    return ckpt.get("step", 0)


# =========================================================================== #
# Step-level helpers (pure forward computations — no optimizer logic)
# =========================================================================== #

def flow_step(model, flow_matcher, x_real):
    """Phase 1 forward: OT flow loss only. Returns scalar loss tensor."""
    x0 = torch.randn_like(x_real)
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x_real)
    vt = model(t, xt)
    mse = (vt - ut).square()
    if FLAGS.use_flow_weight:
        w = flow_weight(t, cutoff=FLAGS.time_cutoff)
        return torch.mean(w * mse.mean(dim=[1, 2, 3]))
    return mse.mean()


def cd_step(model, flow_matcher, x_real_flow, x_real_cd, device):
    """Phase 2 forward: OT flow + Contrastive Divergence.

    Returns (total_loss, flow_loss, cd_loss, pos_energy, neg_energy).
    """
    # --- flow part (reuses phase 1 logic) ---
    f_loss = flow_step(model, flow_matcher, x_real_flow)

    # --- CD part ---
    t_dummy = torch.ones(x_real_cd.size(0), device=device)
    pos_energy = model.potential(x_real_cd, t_dummy)

    B = x_real_cd.size(0)
    if FLAGS.split_negative:
        half = B // 2
        x_neg_init = torch.empty_like(x_real_cd)
        x_neg_init[:half] = x_real_cd[:half]
        x_neg_init[half:] = torch.randn_like(x_neg_init[half:])
        at_data_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if not FLAGS.same_temperature_scheduler:
            at_data_mask[:half] = True
    else:
        x_neg_init = torch.randn_like(x_real_cd)
        at_data_mask = torch.zeros(B, dtype=torch.bool, device=device)

    x_neg = gibbs_sampling_time_sweep(
        x_init=x_neg_init,
        model=model,
        at_data_mask=at_data_mask,
        n_steps=FLAGS.n_gibbs,
        dt=FLAGS.dt_gibbs,
    )
    neg_energy = model.potential(x_neg, t_dummy)

    # optional trimming of highest-energy negatives
    if FLAGS.cd_trim_fraction > 0.0:
        k = int(FLAGS.cd_trim_fraction * B)
        if k > 0:
            neg_sorted, _ = neg_energy.sort()
            neg_stat = neg_sorted[: B - k].mean()
        else:
            neg_stat = neg_energy.mean()
    else:
        neg_stat = neg_energy.mean()

    cd_loss = FLAGS.lambda_cd * (pos_energy.mean() - neg_stat)
    if FLAGS.cd_neg_clamp > 0:
        cd_loss = torch.maximum(
            cd_loss, torch.tensor(-FLAGS.cd_neg_clamp, device=device)
        )

    return f_loss + cd_loss, f_loss, cd_loss, pos_energy, neg_energy


# =========================================================================== #
# Shared training loop
# =========================================================================== #

def _train_loop(
    model,
    ema_model,
    optimizer,
    scheduler,
    step_fn,        # callable(model) -> (loss, log_dict[str -> float])
    datalooper,     # used only for periodic sample generation at save_step
    total_steps,
    start_step,
    ema_decay,
    device,
    savedir,
    phase_tag,
    scaler=None,
    amp_dtype=None,
):
    """
    Owns all boilerplate:
        zero_grad → backward → clip → step → scheduler → ema → log → checkpoint.

    step_fn encapsulates all forward computation; it receives the model and
    returns (scalar loss tensor, dict of float values to log).
    """
    log_every = FLAGS.log_every
    last_log_time = time.time()

    with torch.backends.cuda.sdp_kernel(
        enable_math=True, enable_flash=False, enable_mem_efficient=False
    ):
        for step in range(start_step, start_step + total_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(amp_dtype is not None)):
                loss, log_dict = step_fn(model)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clip)
                optimizer.step()
            scheduler.step()
            ema(model, ema_model, ema_decay)

            if step % log_every == 0:
                now = time.time()
                elapsed = now - last_log_time
                sps = log_every / elapsed if elapsed > 1e-9 else 0.0
                last_log_time = now
                lr = scheduler.get_last_lr()[0]
                stats = "  ".join(f"{k}={v:.5f}" for k, v in log_dict.items())
                logging.info(
                    f"[{phase_tag} step {step}]  {stats}  lr={lr:.6f}  {sps:.2f} it/s"
                )

            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0 and step > start_step:
                _periodic_save(model, ema_model, optimizer, scheduler,
                               datalooper, step, device, savedir, phase_tag)


def _periodic_save(model, ema_model, optimizer, scheduler,
                   datalooper, step, device, savedir, phase_tag):
    """Sample grids + checkpoint written at every save_step."""
    real_batch = next(datalooper).to(device)
    generate_samples(model, savedir, step, net_="normal", real_data=real_batch[:8])
    generate_samples(ema_model, savedir, step, net_="ema", real_data=real_batch[:8])

    # negative samples for pos/neg grid
    x_pos = real_batch[:64]
    x_neg_init = torch.randn_like(x_pos)
    at_data_mask = torch.zeros(x_pos.size(0), dtype=torch.bool, device=device)
    x_neg = gibbs_sampling_time_sweep(
        x_init=x_neg_init,
        model=model,
        at_data_mask=at_data_mask,
        n_steps=FLAGS.n_gibbs,
        dt=FLAGS.dt_gibbs,
    )
    save_pos_neg_grids(x_pos, x_neg, savedir, step)

    save_checkpoint(model, ema_model, optimizer, scheduler, step, savedir,
                    tag=f"{phase_tag}_step{step}")


# =========================================================================== #
# Phase wrappers
# =========================================================================== #

def _resolve_ema_decay(flag_value, n_steps):
    """Return flag_value if explicitly set, else auto-compute exp(-10 / n_steps)."""
    if flag_value >= 0.0:
        return flag_value
    decay = math.exp(-10.0 / n_steps)
    logging.info(f"EMA decay auto-computed: exp(-10 / {n_steps}) = {decay:.6f}")
    return decay


def train_phase1(model, ema_model, optimizer, scheduler,
                 datalooper, flow_matcher, device, savedir,
                 scaler=None, amp_dtype=None):
    """Phase 1 — OT flow matching (Algorithm 1)."""
    logging.info(
        f"=== Phase 1: {FLAGS.phase1_steps} steps, "
        f"ema_decay={_resolve_ema_decay(FLAGS.phase1_ema_decay, FLAGS.phase1_steps):.6f}, "
        f"use_flow_weight={FLAGS.use_flow_weight} ==="
    )

    def step_fn(model):
        x_real = next(datalooper).to(device)
        loss = flow_step(model, flow_matcher, x_real)
        return loss, {"flow": loss.item()}

    _train_loop(
        model, ema_model, optimizer, scheduler,
        step_fn, datalooper,
        total_steps=FLAGS.phase1_steps,
        start_step=0,
        ema_decay=_resolve_ema_decay(FLAGS.phase1_ema_decay, FLAGS.phase1_steps),
        device=device,
        savedir=savedir,
        phase_tag="phase1",
        scaler=scaler,
        amp_dtype=amp_dtype,
    )


def train_phase2(model, ema_model, optimizer, scheduler,
                 datalooper, flow_matcher, device, savedir,
                 scaler=None, amp_dtype=None):
    """Phase 2 — OT flow + Contrastive Divergence (Algorithm 2)."""
    logging.info(
        f"=== Phase 2: {FLAGS.phase2_steps} steps, "
        f"ema_decay={_resolve_ema_decay(FLAGS.phase2_ema_decay, FLAGS.phase2_steps):.6f}, "
        f"lambda_cd={FLAGS.lambda_cd}, n_gibbs={FLAGS.n_gibbs} ==="
    )

    def step_fn(model):
        x_real_flow = next(datalooper).to(device)
        x_real_cd = next(datalooper).to(device)
        total_loss, f_loss, cd_loss, pos_e, neg_e = cd_step(
            model, flow_matcher, x_real_flow, x_real_cd, device
        )
        log = {
            "flow":    f_loss.item(),
            "cd":      cd_loss.item(),
            "pos_min": pos_e.min().item(),
            "pos_max": pos_e.max().item(),
            "pos_std": pos_e.std().item(),
            "neg_min": neg_e.min().item(),
            "neg_max": neg_e.max().item(),
            "neg_std": neg_e.std().item(),
        }
        if FLAGS.model_type == "hopfield" and hasattr(model, "count_active_memories"):
            n_per, n_total = model.count_active_memories(x_real_cd)
            log["mem_per_sample"] = n_per
            log["mem_active"] = float(n_total)
        return total_loss, log

    _train_loop(
        model, ema_model, optimizer, scheduler,
        step_fn, datalooper,
        total_steps=FLAGS.phase2_steps,
        start_step=FLAGS.phase1_steps,
        ema_decay=_resolve_ema_decay(FLAGS.phase2_ema_decay, FLAGS.phase2_steps),
        device=device,
        savedir=savedir,
        phase_tag="phase2",
        scaler=scaler,
        amp_dtype=amp_dtype,
    )


# =========================================================================== #
# Main
# =========================================================================== #

def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    if FLAGS.use_amp and device.type == "cuda":
        try:
            _bf16_ok = torch.cuda.is_bf16_supported()
        except RuntimeError:
            _bf16_ok = False
        if _bf16_ok:
            amp_dtype = torch.bfloat16
            scaler = None
            _amp_label = "bfloat16 (no GradScaler)"
        else:
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
            _amp_label = "float16 + GradScaler"
    else:
        amp_dtype = None
        scaler = None
        _amp_label = "disabled"

    savedir = create_timestamped_dir(FLAGS.output_dir, FLAGS.model)
    log_dir = FLAGS.my_log_dir if FLAGS.my_log_dir else savedir
    logging.get_absl_handler().use_absl_log_file(program_name="train", log_dir=log_dir)
    logging.set_verbosity(logging.INFO)
    logging.info(f"Output directory : {savedir}")
    logging.info(f"Device           : {device}")
    logging.info(f"AMP              : {_amp_label}")
    for k, v in FLAGS.flag_values_dict().items():
        logging.info(f"  {k} = {v}")

    dataset = get_dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    datalooper = infiniteloop(dataloader)

    model, ema_model = build_model(device)
    optimizer, scheduler = build_optimizer(model, FLAGS.phase1_steps)
    load_checkpoint(model, ema_model, optimizer, scheduler, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {n_params:,}")

    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    # ------------------------------------------------------------------ #
    # Phase 1
    # ------------------------------------------------------------------ #
    if not FLAGS.skip_phase1:
        train_phase1(model, ema_model, optimizer, scheduler,
                     datalooper, flow_matcher, device, savedir,
                     scaler=scaler, amp_dtype=amp_dtype)
        save_checkpoint(model, ema_model, optimizer, scheduler,
                        FLAGS.phase1_steps, savedir, tag="phase1_final")
    else:
        logging.info("Phase 1 skipped (--skip_phase1).")

    # ------------------------------------------------------------------ #
    # Phase 2
    # ------------------------------------------------------------------ #
    if not FLAGS.skip_phase2:
        # Reset optimizer LR for phase 2 — phase 1 cosine has decayed to ~0.
        # Must set both 'lr' and 'initial_lr': LambdaLR reads initial_lr as
        # its base_lr and would otherwise inherit the phase 1 peak (2e-3).
        for pg in optimizer.param_groups:
            pg['lr'] = FLAGS.phase2_lr
            pg['initial_lr'] = FLAGS.phase2_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0
        )
        logging.info(f"Phase 2 optimizer reset: lr={FLAGS.phase2_lr}")
        train_phase2(model, ema_model, optimizer, scheduler,
                     datalooper, flow_matcher, device, savedir,
                     scaler=scaler, amp_dtype=amp_dtype)
        save_checkpoint(model, ema_model, optimizer, scheduler,
                        FLAGS.phase1_steps + FLAGS.phase2_steps,
                        savedir, tag="final")

        # ---------------------------------------------------------------- #
        # FID
        # ---------------------------------------------------------------- #
        if FLAGS.fid_num_gen > 0:
            from fid import compute_fid
            logging.info("=== FID evaluation on final EMA model ===")
            del optimizer, scheduler, model
            torch.cuda.empty_cache()
            compute_fid(ema_model, FLAGS, device, savedir)
        else:
            logging.info("FID skipped (--fid_num_gen=0).")
    else:
        logging.info("Phase 2 and FID skipped (--skip_phase2).")


if __name__ == "__main__":
    app.run(main)
