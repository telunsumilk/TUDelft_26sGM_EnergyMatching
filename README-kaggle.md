# Running on Kaggle GPU

You can also use the free kaggle gpu (you need a verified kaggle account for that), for example for the evaluation task. 
Upload the repo ZIP as a private Kaggle Dataset, then add it as an input to a notebook.

**Make sure you enabled the acceleration for this notebook in the settings**

The ZIP should include the repo code and the checkpoint:

```text
experiments/
utils_cifar_imagenet.py
requirements.txt
checkpoints/
```

Do not include `.git/`, `.venv/`, `sampling_results/`, `metric_results/`, or other old outputs.

## 1. Copy Repo Into Working Directory

In the first notebook cell, copy the mounted dataset into writable storage and install requirements:

```bash
!mkdir -p /kaggle/working/energy-matching
!cp -r /kaggle/input/datasets/<YOUR_USERNAME>/<YOUR_REPO_NAME>/* /kaggle/working/energy-matching/.
%cd /kaggle/working/energy-matching
!pip install -r requirements.txt
```

The repo dataset should include the checkpoint for now:

```text
checkpoints/cifar10_main_training_147000.pt
```

## 2. Generate CIFAR-10 Images

Run generation in a separate cell:

```bash
!python3 experiments/cifar10/generate_cifar_dataset.py \
  --resume_ckpt=checkpoints/cifar10_main_training_147000.pt \
  --num_samples=1000 \
  --batch_size=16 \
  --time_cutoff=1.0 \
  --epsilon_max=0.01 \
  --dt_gibbs=0.01 \
  --t_end=3.25 \
  --use_ema=True \
  --progress_chunk_steps=10 \
  --output_dir=/kaggle/working/sampling_results
```

If Kaggle runs out of GPU memory, lower `--batch_size`.

## 3. Evaluate FID

Run evaluation on the generated image folder:

```bash
!python3 experiments/cifar10/evaluate_cifar_generated.py \
  --generated_dir=/kaggle/working/sampling_results/cifar10_em_<timestamp>/images \
  --batch_size=128 \
  --num_workers=4 \
  --output_dir=/kaggle/working/metric_results
```

Replace `cifar10_em_<timestamp>` with the folder created by the generation script.

You can also download `/kaggle/working/sampling_results` and run the FID/KID evaluation locally later.
