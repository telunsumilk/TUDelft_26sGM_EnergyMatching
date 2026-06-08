# config.py — flag definitions shared by train.py and fid.py (standalone).
from absl import flags

_defined = False


def define_flags():
    global _defined
    if _defined:
        return
    _defined = True

    # ------------------------------------------------------------------ #
    # Model identity / output
    # ------------------------------------------------------------------ #
    flags.DEFINE_string("model", "genmodel", "Model name prefix for timestamped output dir.")
    flags.DEFINE_string("output_dir", "../../results/", "Root directory for run outputs.")
    flags.DEFINE_string("my_log_dir", "", "Override directory for absl log files.")
    flags.DEFINE_enum("model_type", "vit", ["vit", "attn", "mlp", "hopfield", "cnn", "cnn_hopfield"],
                      "Head architecture: vit (full Transformer), attn (single MHA layer), mlp (global-pool MLP), "
                      "hopfield (Hopfield energy head on UNet backbone), cnn (lightweight encoder, no UNet).")

    # ------------------------------------------------------------------ #
    # UNet architecture
    # ------------------------------------------------------------------ #
    flags.DEFINE_integer("num_channels", 128, "Base channel count for UNet.")
    flags.DEFINE_integer("num_res_blocks", 2, "Residual blocks per resolution stage.")
    flags.DEFINE_list("channel_mult", ["1", "2", "2", "2"],
                      "Channel multipliers per UNet resolution level.")
    flags.DEFINE_string("attention_resolutions", "16",
                        "Comma-separated resolutions at which UNet uses self-attention.")
    flags.DEFINE_integer("num_heads", 4, "Attention heads inside UNet.")
    flags.DEFINE_integer("num_head_channels", 64, "Channels per UNet attention head.")
    flags.DEFINE_float("dropout", 0.1, "Dropout rate in UNet and Transformer layers.")
    flags.DEFINE_float("output_scale", 1000.0, "Scalar multiplier for final potential output.")
    flags.DEFINE_float("energy_clamp", None,
                       "Tanh-based clamp magnitude for potential (None = disabled). "
                       "ImageNet32 default: 10000.")

    # ------------------------------------------------------------------ #
    # ViT head architecture
    # ------------------------------------------------------------------ #
    flags.DEFINE_integer("embed_dim", 384, "Token embedding dimension for the ViT head.")
    flags.DEFINE_integer("transformer_nheads", 4, "Attention heads in ViT encoder.")
    flags.DEFINE_integer("transformer_nlayers", 8, "Transformer encoder depth.")

    # ------------------------------------------------------------------ #
    # Hopfield head (model_type=hopfield only)
    # ------------------------------------------------------------------ #
    flags.DEFINE_integer("hopfield_memories", 512, "Number of Hopfield memory prototypes.")
    flags.DEFINE_float("hopfield_beta", 8.0, "Hopfield inverse temperature β.")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    flags.DEFINE_string("dataset", "cifar10", "Dataset to use: cifar10 | imagenet32.")
    flags.DEFINE_float("color_jitter", 0.0,
                       "Color jitter strength for brightness/contrast/saturation (hue = strength/4). "
                       "0 = disabled. Typical: 0.2 (mild) to 0.4 (strong).")
    flags.DEFINE_list("cifar_classes", [],
                      "Comma-separated CIFAR-10 class indices (0-9) to train on. "
                      "Empty = all 10 classes. "
                      "0=airplane,1=automobile,2=bird,3=cat,4=deer,"
                      "5=dog,6=frog,7=horse,8=ship,9=truck. "
                      "Example: 3,4,5,7 for 4-legged animals.")
    flags.DEFINE_list("imagenet_classes", [],
                      "Comma-separated ImageNet class indices (0-999) to train on. "
                      "Empty = all classes. Example: 281,282,283,284,285 for cat breeds.")

    # ------------------------------------------------------------------ #
    # Optimizer / scheduler
    # ------------------------------------------------------------------ #
    flags.DEFINE_float("lr", 1.2e-3, "Peak learning rate. (ImageNet32 default: 6e-4)")
    flags.DEFINE_float("grad_clip", 1.0, "Gradient norm clipping threshold.")
    flags.DEFINE_integer("warmup", 10000, "Linear LR warmup steps.")
    flags.DEFINE_integer("batch_size", 128, "Per-GPU batch size.")
    flags.DEFINE_integer("fid_batch_size", 256,
                         "Batch size for FID (real loader + fake generation). "
                         "UNet skip connections scale with batch size — keep ≤256 on 24 GB GPUs.")
    flags.DEFINE_integer("num_workers", 4, "DataLoader worker processes.")

    # ------------------------------------------------------------------ #
    # Phase 1 (OT flow matching)
    # ------------------------------------------------------------------ #
    flags.DEFINE_integer("phase1_steps", 145000,
                         "Phase 1 training steps. (ImageNet32 default: 640000)")
    flags.DEFINE_float("phase1_ema_decay", -1.0,
                     "EMA decay factor during Phase 1. "
                     "-1 = auto: exp(-10 / phase1_steps), e.g. 0.9993 at 15K, 0.99993 at 145K.")
    flags.DEFINE_bool("use_flow_weight", True,
                      "Apply time-dependent flow loss weighting. Set False for ImageNet32.")
    flags.DEFINE_float("time_cutoff", 1.0,
                       "flow_weight decays linearly to 0 between time_cutoff and 1.0.")

    # ------------------------------------------------------------------ #
    # Phase 2 (OT flow + Contrastive Divergence)
    # ------------------------------------------------------------------ #
    flags.DEFINE_integer("phase2_steps", 2000, "Phase 2 training steps.")
    flags.DEFINE_float("phase2_ema_decay", -1.0,
                     "EMA decay factor during Phase 2. "
                     "-1 = auto: exp(-10 / phase2_steps).")
    flags.DEFINE_float("lambda_cd", 0.0, "CD loss coefficient (0 = Phase 1 only).")
    flags.DEFINE_integer("n_gibbs", 0, "MCMC steps for negative sample generation.")
    flags.DEFINE_float("dt_gibbs", 0.01, "Step size for Gibbs / SDE sampling.")
    flags.DEFINE_float("epsilon_max", 0.0,
                       "Maximum noise magnitude for plot_epsilon SDE schedule.")
    flags.DEFINE_float("cd_neg_clamp", 0.02,
                       "Clamp CD loss from below at -cd_neg_clamp (0 = disabled).")
    flags.DEFINE_float("cd_trim_fraction", 0.1,
                       "Fraction of highest neg-energy negatives to discard when computing CD.")
    flags.DEFINE_bool("split_negative", False,
                      "Init half negatives from x_real_cd, half from Gaussian noise.")
    flags.DEFINE_bool("same_temperature_scheduler", True,
                      "Use same MCMC temperature schedule for all negatives (ignore at_data_mask).")

    # ------------------------------------------------------------------ #
    # Checkpointing / logging
    # ------------------------------------------------------------------ #
    flags.DEFINE_integer("save_step", 5000,
                         "Save checkpoint and sample grids every N steps (0 = never).")
    flags.DEFINE_integer("log_every", 10, "Log training stats every N steps.")
    flags.DEFINE_string("resume_ckpt", "", "Path to checkpoint to resume from.")
    flags.DEFINE_bool("skip_phase1", False,
                      "Skip Phase 1 (useful when resuming from a phase1_final checkpoint).")
    flags.DEFINE_bool("skip_phase2", False, "Skip Phase 2 and FID computation.")

    # ------------------------------------------------------------------ #
    # FID evaluation
    # ------------------------------------------------------------------ #
    flags.DEFINE_list(
        "fid_times",
        [str(t) for t in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                           3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]],
        "SDE end-times for the FID sweep. (ImageNet32 default: 0.75,1.0,...,4.0)"
    )
    flags.DEFINE_integer("fid_num_gen", 50000,
                         "Number of fake images to generate for FID (0 = skip FID).")

    # ------------------------------------------------------------------ #
    # Performance
    # ------------------------------------------------------------------ #
    flags.DEFINE_bool("use_amp", True,
                      "Enable automatic mixed precision (bf16 if available, else fp16). "
                      "No-op on CPU.")

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    flags.DEFINE_bool("debug", False, "Enable debug mode (shorter runs, extra logging).")


def parse_channel_mult(FLAGS):
    return [int(c) for c in FLAGS.channel_mult]
