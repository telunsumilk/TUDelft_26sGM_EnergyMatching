# Unconditional CIFAR-10 Generation

For now we use the pretrained CIFAR-10 Energy Matching checkpoint from Hugging Face:
`m1balcerak/energy_matching/cifar10_main_training_147000.pt`.
Later this should be replaced by our own trained checkpoint.

## Generate Images

Use `experiments/cifar10/generate_cifar_dataset.py` to generate and save individual images.

```bash
python3 experiments/cifar10/generate_cifar_dataset.py \
  --resume_ckpt=checkpoints/cifar10_main_training_147000.pt \
  --num_samples=1000 \
  --batch_size=16 \
  --time_cutoff=1.0 \
  --epsilon_max=0.01 \
  --dt_gibbs=0.01 \
  --t_end=3.25 \
  --use_ema=True \
  --progress_chunk_steps=10 \
  --output_dir=./sampling_results
```

Output:

```text
sampling_results/cifar10_em_<timestamp>/
  config.json
  images/
    000000.png
    000001.png
    ...
```

Important flags:
- `--num_samples`: number of images to generate.
- `--batch_size`: generation batch size; lower if GPU OOMs.
- `--t_end`: sampling end time. Paper-style CIFAR uses `3.25`.
- `--dt_gibbs`: SDE step size. Paper-style uses `0.01`.
- `--progress_chunk_steps`: tqdm update frequency; use `0` for original one-shot SDE solve.

## Evaluate FID

Use `experiments/cifar10/evaluate_cifar_generated.py` to compute FID from a generated image folder.

```bash
python3 experiments/cifar10/evaluate_cifar_generated.py \
  --generated_dir=./sampling_results/cifar10_em_<timestamp>/images \
  --batch_size=128 \
  --num_workers=8 \
  --output_dir=./metric_results
```

This compares the saved generated images against CIFAR-10 real images and writes a JSON result to `metric_results/`.

Small generated sets are only useful for checking the pipeline. For meaningful FID/KID, generate at least thousands of images.
