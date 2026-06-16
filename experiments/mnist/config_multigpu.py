# File: config_multigpu.py

from absl import flags


def define_flags():
    FLAGS = flags.FLAGS

    # Model + output
    flags.DEFINE_string("model", "EBMTime", "Flow matching model type")
    flags.DEFINE_string("output_dir", "./results/", "Directory for results")

    # Flow/EBM model parameters. Defaults match the MNIST checkpoint logs.
    flags.DEFINE_integer("num_channels", 32, "Base channels")
    flags.DEFINE_integer("num_res_blocks", 2, "Number of resblocks per stage")
    flags.DEFINE_float(
        "energy_clamp",
        None,
        "Energy clamp (tanh-based). If None, no clamp is applied.",
    )

    # UNet + attention
    flags.DEFINE_integer("num_heads", 2, "Number of attention heads.")
    flags.DEFINE_integer("num_head_channels", 64, "Number of channels per attention head.")
    flags.DEFINE_float("dropout", 0.1, "Dropout rate.")
    flags.DEFINE_string("attention_resolutions", "16", "Attention at these resolution(s).")

    # Patch-based ViT parameters
    flags.DEFINE_integer("embed_dim", 128, "Embedding dimension for patch-based ViT head.")
    flags.DEFINE_integer("transformer_nheads", 2, "Number of heads in the ViT encoder.")
    flags.DEFINE_integer("transformer_nlayers", 2, "Number of layers in the ViT encoder.")
    flags.DEFINE_float("output_scale", 100.0, "Multiplier for final potential output.")
    flags.DEFINE_list(
        "channel_mult",
        ["1", "2", "2"],
        "Channel multipliers for each UNet resolution block.",
    )

    flags.DEFINE_bool("debug", False, "Debug mode")

    # Training flags kept for compatibility with the CIFAR config surface.
    flags.DEFINE_float("lr", 1.2e-3, "Learning rate")
    flags.DEFINE_float("grad_clip", 1.0, "Gradient norm clipping")
    flags.DEFINE_integer("total_steps", 500001, "Total training steps")
    flags.DEFINE_integer("warmup", 10000, "Learning rate warmup steps")
    flags.DEFINE_integer("batch_size", 128, "Batch size")
    flags.DEFINE_integer("num_workers", 4, "Dataloader workers")
    flags.DEFINE_float("ema_decay", 0.999, "EMA decay")

    # Evaluation / saving
    flags.DEFINE_integer("save_step", 5000, "Checkpoint save frequency (0=disable)")
    flags.DEFINE_string("resume_ckpt", "", "Path to checkpoint for resuming training")

    # EBM + CD
    flags.DEFINE_float("epsilon_max", 0.0, "Max step size in Gibbs sampling")
    flags.DEFINE_float("dt_gibbs", 0.01, "Step size for Gibbs sampling")
    flags.DEFINE_integer("n_gibbs", 0, "Number of Gibbs steps")
    flags.DEFINE_float("lambda_cd", 0.0, "Coefficient for contrastive divergence loss")
    flags.DEFINE_float("time_cutoff", 1.0, "Flow loss decays to zero beyond t>=time_cutoff")
    flags.DEFINE_float(
        "cd_neg_clamp",
        0.02,
        "Clamp negative total CD below -cd_neg_clamp. 0=disable clamp.",
    )
    flags.DEFINE_float(
        "cd_trim_fraction",
        0.1,
        "Fraction of highest negative energies discarded for CD (0=disable).",
    )
    flags.DEFINE_bool(
        "split_negative",
        False,
        "If True, initialize half of the negative samples from x_real_cd, half from noise",
    )
    flags.DEFINE_bool(
        "same_temperature_scheduler",
        True,
        "If True, ignore at_data_mask and use the same temperature schedule for all samples",
    )

    flags.DEFINE_string("my_log_dir", "", "Directory for Abseil logs.")


def parse_channel_mult(FLAGS):
    return [int(c) for c in FLAGS.channel_mult]
