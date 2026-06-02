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
     `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
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
curl -fsSL https://raw.githubusercontent.com/m1balcerak/EnergyMatching/main/setup.sh \
  | bash -s -- /root
```

Or if you have already cloned locally, sync with rsync instead:

```bash
rsync -avz --exclude='.git' --exclude='results/' --exclude='.venv/' \
  /path/to/EnergyMatching/ \
  -e "ssh -p 12345" root@123.45.67.89:/root/EnergyMatching/
```

Then on the instance:

```bash
bash /root/EnergyMatching/setup.sh /root
```

---

## 4. (ImageNet32 only) Upload the dataset

The CIFAR-10 dataset downloads automatically on first run.
ImageNet32 must be uploaded manually:

```bash
# From your local machine — upload the Imagenet32_train folder
rsync -avz --progress \
  /path/to/Imagenet32_train/ \
  -e "ssh -p 12345" root@123.45.67.89:/root/EnergyMatching/experiments/genmodel/data/Imagenet32_train/
```

---

## 5. Run training

```bash
# On the instance, inside the tmux session
source /root/venv/bin/activate
cd /root/EnergyMatching/experiments/genmodel

# CIFAR-10 (full run)
python train.py \
  --dataset=cifar10 \
  --phase1_steps=145000 \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01 \
  --fid_num_gen=50000 \
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
  --fid_times=0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --fid_num_gen=50000 \
  --batch_size=64
```

Results are written to `../../results/` (i.e. `/root/EnergyMatching/results/`).

Detach from tmux with **Ctrl-B D** and re-attach later with `tmux attach -t train`.

---

## 6. Monitor progress

```bash
# Tail the absl log (filename matches the run timestamp)
tail -f /root/EnergyMatching/results/genmodel_*/train.INFO
```

---

## 7. Download results before the instance stops

vast.ai instances have **no persistent storage** — all data is lost when the
instance is destroyed. Download checkpoints and results regularly:

```bash
# From your local machine
rsync -avz --progress \
  -e "ssh -p 12345" \
  root@123.45.67.89:/root/EnergyMatching/results/ \
  ./results_vastai/
```

Or use `nohup rsync ...` on the instance to push to your own server.

---

## 8. Resume from a checkpoint

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
