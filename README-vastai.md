# Running EnergyMatching on vast.ai

## Recommended GPU

| Dataset | Minimum | Comfortable |
|---------|---------|-------------|
| CIFAR-10 | RTX 3080 16 GB | RTX 3090 / 4090 24 GB |
| ImageNet32 | RTX 3090 24 GB | A100 40/80 GB |

CIFAR-10 training takes ~10 h on an RTX 3090; ImageNet32 ~3–4 days.

---

## 1. Add your SSH key

Go to **Account → SSH Keys** on the vast.ai console and paste your public key
(`~/.ssh/id_rsa.pub` or `id_ed25519.pub`).

---

## 2. Rent an instance

1. Open the **Search** page and filter by:
   - GPU: RTX 3090 or better
   - Disk: ≥ 50 GB (CIFAR-10 ~500 MB; ImageNet32 ~35 GB raw)
   - Image: search for **pytorch** — select an image like
     `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`
2. Click **Rent** and wait for the instance to start.
3. Copy the SSH command shown in the console — it looks like:
   ```
   ssh -p 12345 root@123.45.67.89
   ```

---

## 3. Connect and set up

```bash
# Connect
ssh -p 12345 root@123.45.67.89

# Start a tmux session so training survives disconnections
tmux new -s train

# Download and run the setup script
curl -fsSL https://raw.githubusercontent.com/telunsumilk/TUDelft_26sGM_EnergyMatching/refs/heads/main/setup.sh \
  | bash
```

Or if you have already cloned locally, sync with rsync instead:

```bash
rsync -avz --exclude='.git' --exclude='results/' --exclude='.venv/' \
  /path/to/EnergyMatching/ \
  -e "ssh -p 12345" root@123.45.67.89:/workspace/EnergyMatching/
```

Then on the instance:

```bash
bash /workspace/EnergyMatching/setup.sh
```

---

## 4. (ImageNet32 only) Upload the dataset

The CIFAR-10 dataset downloads automatically on first run.
ImageNet32 must be uploaded manually:

```bash
# From your local machine — upload the Imagenet32_train folder
rsync -avz --progress \
  /path/to/Imagenet32_train/ \
  -e "ssh -p 12345" root@123.45.67.89:/workspace/EnergyMatching/experiments/genmodel/data/Imagenet32_train/
```

---

## 5. Run training

```bash
# On the instance, inside the tmux session
source /workspace/venv/bin/activate
cd /workspace/EnergyMatching/experiments/genmodel

# CIFAR-10 (full run)
python train.py \
  --dataset=cifar10 \
  --phase1_steps=145000 \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01 \
  --fid_num_gen=50000 \
  --fid_times=1.0,2.0,3.0,4.0,5.0 \
  --batch_size=128

# CIFAR-10 — subset of classes (faster, lower memory)
# Class indices: 0=airplane,1=automobile,2=bird,3=cat,4=deer,
#                5=dog,6=frog,7=horse,8=ship,9=truck
# Omit --cifar_classes to train on all 10 classes.
python train.py \
  --dataset=cifar10 \
  --cifar_classes=3,4,5,7 \
  --phase1_steps=145000 \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01 \
  --fid_num_gen=20000 \
  --fid_times=1.0,2.0,3.0,4.0,5.0 \
  --batch_size=128

# ImageNet32 (key overrides)
python train.py \
  --dataset=imagenet32 \
  --lr=6e-4 \
  --energy_clamp=10000 \
  --use_flow_weight=False \
  --phase1_steps=640000 \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01 \
  --fid_times=0.75,1.5,2.5,3.5,4.0 \
  --fid_num_gen=50000 \
  --batch_size=64

# ImageNet32 — subset of classes (faster, lower memory)
# Class indices are 0-based (0–999). Example: cat breeds (281–285).
# Omit --imagenet_classes to train on all 1000 classes.
python train.py \
  --dataset=imagenet32 \
  --imagenet_classes=281,282,283,284,285 \
  --lr=6e-4 \
  --energy_clamp=10000 \
  --use_flow_weight=False \
  --phase1_steps=640000 \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01 \
  --fid_times=0.75,1.5,2.5,3.5,4.0 \
  --fid_num_gen=50000 \
  --batch_size=64

# Model architecture — five heads are available via --model_type:
#   vit       Full Transformer head (default)
#   attn      Single self-attention layer (lighter)
#   mlp       Global-pool MLP head (lightest UNet-backed option)
#   hopfield  Modern Hopfield energy head on UNet backbone;
#             energy wells form at learned prototypes (--hopfield_memories,
#             --hopfield_beta control the number and sharpness of wells)
#   cnn       Lightweight pure encoder — no UNet decoder or skip connections;
#             ~4-8× fewer parameters, faster to train
# Examples:
python train.py \
  --dataset=cifar10 \
  --model_type=attn \
  --phase1_steps=145000 \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01 \
  --fid_num_gen=50000 \
  --fid_times=1.0,2.0,3.0,4.0,5.0 \
  --batch_size=128

python train.py \
  --dataset=cifar10 \
  --model_type=hopfield \
  --hopfield_memories=512 \
  --hopfield_beta=8.0 \
  --phase1_steps=145000 \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01 \
  --fid_num_gen=50000 \
  --fid_times=1.0,2.0,3.0,4.0,5.0 \
  --batch_size=128

python train.py \
  --dataset=cifar10 \
  --model_type=cnn \
  --phase1_steps=145000 \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01 \
  --fid_num_gen=50000 \
  --fid_times=1.0,2.0,3.0,4.0,5.0 \
  --batch_size=128
```

Results are written to `../../results/` (i.e. `/workspace/EnergyMatching/results/`).

Detach from tmux with **Ctrl-B D** and re-attach later with `tmux attach -t train`.

---

## 6. Monitor progress

```bash
# Tail the absl log (filename matches the run timestamp)
tail -f /workspace/EnergyMatching/results/genmodel_*/train.INFO
```

---

## 7. Download results before the instance stops

vast.ai instances have **no persistent storage** — all data is lost when the
instance is destroyed. Download checkpoints and results regularly:

```bash
# From your local machine
rsync -avz --progress \
  -e "ssh -p 12345" \
  root@123.45.67.89:/workspace/EnergyMatching/results/ \
  ./results_vastai/
```

Or use `nohup rsync ...` on the instance to push to your own server.

---

## 8. Run inpainting

After training, run the inpainting demo on CIFAR-10 test images:

```bash
source /workspace/venv/bin/activate
cd /workspace/EnergyMatching/experiments/genmodel

python inpainting.py \
  --checkpoint ../../results/genmodel_TIMESTAMP/cifar10_checkpoint_phase1_final.pt \
  --num_test_images 1 \
  --mask_type center \
  --num_chains 2 \
  --n_inpaint_steps 300 \
  --inpaint_savedir results/inpainting
```

Results are saved to `results/inpainting/inpaint_0000.png` as a grid:
`[original | masked input | chain 0 | chain 1 | ...]`

To enable the interaction energy (encourages diverse completions):

```bash
python inpainting.py \
  --checkpoint ../../results/genmodel_TIMESTAMP/cifar10_checkpoint_phase1_final.pt \
  --num_test_images 1 \
  --mask_type center \
  --num_chains 4 \
  --n_inpaint_steps 300 \
  --interaction_sigma 0.5 \
  --interaction_mask_fraction 0.5 \
  --inpaint_savedir results/inpainting
```

`--interaction_mask_fraction` controls which sub-region of the mask drives diversity:
`1.0` = full inpaint mask, `0.5` = inner half of its bounding box, `0.0` = disabled.

---

## 9. Resume from a checkpoint

If the instance is interrupted, re-upload the checkpoint and resume:

```bash
python train.py \
  --dataset=cifar10 \
  --skip_phase1 \
  --resume_ckpt=../../results/genmodel_TIMESTAMP/cifar10_checkpoint_phase1_final.pt \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01
```
