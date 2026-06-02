# Running EnergyMatching on RunPod

## Recommended GPU

| Dataset | Minimum | Comfortable |
|---------|---------|-------------|
| CIFAR-10 | RTX 3080 16 GB | RTX 3090 / 4090 24 GB |
| ImageNet32 | RTX 3090 24 GB | A100 40/80 GB |

CIFAR-10 training takes ~10 h on an RTX 3090; ImageNet32 ~3–4 days.

---

## Directory layout on the pod

Everything lives under `/workspace` (the Network Volume):

```
/workspace/
├── EnergyMatching/          # git repo
│   ├── experiments/genmodel/  # training code
│   └── results/             # checkpoints + FID outputs (written here automatically)
└── venv/                    # Python virtual environment
```

---

## 1. (Recommended) Create a Network Volume

A Network Volume persists across pod restarts — use it for the dataset and
checkpoints so you never lose work when a pod is stopped.

1. Go to **Storage → + Network Volume**
2. Name it (e.g. `energymatching-data`), set size ≥ 50 GB for CIFAR-10,
   ≥ 100 GB for ImageNet32
3. Select the same datacenter region you will use for the pod

---

## 2. Launch a pod

1. Go to **Pods → + Deploy**
2. Select a GPU (RTX 3090 or better)
3. Under **Template**, choose a RunPod PyTorch template, e.g.
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
4. Under **Volume**, attach the Network Volume created above → mount at
   `/workspace`
5. Enable **SSH** access
6. Click **Deploy**

---

## 3. Connect

Once the pod is running, click **Connect** to get the SSH command:

```bash
ssh root@<POD_IP> -p <PORT>
```

Or use the web terminal from the RunPod console.

---

## 4. Set up the environment

```bash
# Start a tmux session so training survives disconnections
tmux new -s train

# Download and run the setup script — installs repo + venv under /workspace
curl -fsSL https://raw.githubusercontent.com/telunsumilk/TUDelft_26sGM_EnergyMatching/refs/heads/main/setup.sh \
  | bash -s -- /workspace
```

This creates `/workspace/EnergyMatching/` and `/workspace/venv/`.

Or sync from your local machine first:

```bash
# From your local machine
rsync -avz --exclude='.git' --exclude='results/' --exclude='.venv/' \
  /path/to/EnergyMatching/ \
  root@<POD_IP>:/workspace/EnergyMatching/

# Then on the pod
bash /workspace/EnergyMatching/setup.sh /workspace
```

---

## 5. (ImageNet32 only) Upload the dataset

The CIFAR-10 dataset downloads automatically on first run.
ImageNet32 must be uploaded once to the Network Volume (it persists there):

```bash
# From your local machine
rsync -avz --progress \
  /path/to/Imagenet32_train/ \
  root@<POD_IP>:/workspace/EnergyMatching/experiments/genmodel/data/Imagenet32_train/
```

Because `/workspace` is a Network Volume, you only need to do this once — the
data is available for every future pod that mounts the same volume.

---

## 6. Run training

```bash
# On the pod, inside the tmux session
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
  --fid_times=0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --fid_num_gen=50000 \
  --batch_size=64
```

Results are written to `/workspace/EnergyMatching/results/`.
Because `/workspace` is a Network Volume, checkpoints persist even if you stop
or rebuild the pod.

Detach from tmux with **Ctrl-B D** and re-attach later with `tmux attach -t train`.

---

## 7. Monitor progress

```bash
tail -f /workspace/EnergyMatching/results/genmodel_*/train.INFO
```

---

## 8. Download results

```bash
# From your local machine
rsync -avz --progress \
  root@<POD_IP>:/workspace/EnergyMatching/results/ \
  ./results_runpod/
```

Because results live on the Network Volume, you can also stop the pod to save
cost and re-attach it later to a cheaper pod just for downloading.

---

## 9. Resume from a checkpoint

```bash
source /workspace/venv/bin/activate
cd /workspace/EnergyMatching/experiments/genmodel

python train.py \
  --dataset=cifar10 \
  --skip_phase1 \
  --resume_ckpt=/workspace/EnergyMatching/results/genmodel_TIMESTAMP/cifar10_checkpoint_phase1_final.pt \
  --phase2_steps=2000 \
  --n_gibbs=200 \
  --lambda_cd=1e-3 \
  --epsilon_max=0.01
```

---

## Cost tip

Once Phase 1 is done and the checkpoint is saved to the Network Volume, you can
**stop the pod** and start a new (possibly cheaper) one to run Phase 2 and FID,
since the checkpoint persists.
