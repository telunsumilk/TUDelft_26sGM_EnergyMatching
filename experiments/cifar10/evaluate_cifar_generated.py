###############################################################################
# File: evaluate_cifar_generated.py
#
# Purpose:
#   Compute image quality metrics for an existing folder of generated CIFAR-10
#   samples. This does not generate images; it evaluates PNGs saved by
#   generate_cifar_dataset.py or any compatible script.
#
# Example:
#   python experiments/cifar10/evaluate_cifar_generated.py \
#       --generated_dir=./sampling_results/cifar10_em_20260604_120000/images \
#       --batch_size=128 \
#       --num_workers=4
###############################################################################
import json
import os
from datetime import datetime

import torch
import torchvision.transforms as T
from absl import app, flags, logging
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "generated_dir",
    None,
    "Directory containing generated PNG/JPG/WebP images.",
)
flags.DEFINE_string(
    "output_dir",
    "./metric_results",
    "Directory where metric summary JSON will be saved.",
)
flags.DEFINE_string(
    "cifar10_path",
    None,
    "CIFAR-10 root directory. Defaults to CIFAR10_PATH env var or ./data.",
)
flags.DEFINE_bool(
    "download_cifar10",
    True,
    "Download CIFAR-10 if it is not already present.",
)
flags.DEFINE_bool(
    "use_train_split",
    True,
    "Use CIFAR-10 train split as the real distribution. If False, use test split.",
)
flags.DEFINE_integer("batch_size", 128, "Batch size for metric updates.")
flags.DEFINE_integer("num_workers", 4, "DataLoader workers.")
flags.DEFINE_integer(
    "max_samples",
    0,
    "Optional cap on both real and fake samples. 0 means use all generated images.",
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageFolderFlat(Dataset):
    def __init__(self, root):
        self.root = root
        self.paths = [
            os.path.join(root, name)
            for name in sorted(os.listdir(root))
            if os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No image files found in {root!r}.")
        self.transform = T.Compose(
            [
                T.Resize((32, 32)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            img = img.convert("RGB")
            return self.transform(img)


def to_uint8(images):
    return (images * 255.0).clamp(0, 255).to(torch.uint8)


def limited_batches(loader, max_samples):
    seen = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        if max_samples > 0:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            images = images[:remaining]

        seen += images.shape[0]
        yield images


def main(_):
    if not FLAGS.generated_dir:
        raise ValueError("--generated_dir is required.")
    if not os.path.isdir(FLAGS.generated_dir):
        raise FileNotFoundError(f"--generated_dir not found: {FLAGS.generated_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    generated_dataset = ImageFolderFlat(FLAGS.generated_dir)
    n_fake_total = len(generated_dataset)
    max_samples = FLAGS.max_samples if FLAGS.max_samples > 0 else n_fake_total
    max_samples = min(max_samples, n_fake_total)

    fake_loader = DataLoader(
        generated_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        shuffle=False,
        drop_last=False,
    )

    cifar_root = FLAGS.cifar10_path or os.environ.get("CIFAR10_PATH", "./data")
    real_dataset = __import__("torchvision").datasets.CIFAR10(
        root=cifar_root,
        train=FLAGS.use_train_split,
        download=FLAGS.download_cifar10,
        transform=T.ToTensor(),
    )
    real_loader = DataLoader(
        real_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        shuffle=False,
        drop_last=False,
    )

    n_eval = min(max_samples, len(real_dataset))
    logging.info(f"Evaluating FID and KID with {n_eval} real and {n_eval} generated images.")

    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=min(50, n_eval)).to(device)

    for images in tqdm(limited_batches(real_loader, n_eval), desc="Real images", unit="batch"):
        uint8_images = to_uint8(images.to(device))
        fid.update(uint8_images, real=True)
        kid.update(uint8_images, real=True)

    for images in tqdm(limited_batches(fake_loader, n_eval), desc="Generated images", unit="batch"):
        uint8_images = to_uint8(images.to(device))
        fid.update(uint8_images, real=False)
        kid.update(uint8_images, real=False)

    fid_score = float(fid.compute().detach().cpu())
    kid_mean, kid_std = kid.compute()
    kid_mean = float(kid_mean.detach().cpu())
    kid_std = float(kid_std.detach().cpu())
    logging.info(f"FID = {fid_score:.4f}")
    logging.info(f"KID = {kid_mean:.4f} ± {kid_std:.4f}")

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    summary = {
        "metric": "FID/KID",
        "fid": fid_score,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "generated_dir": FLAGS.generated_dir,
        "num_generated_available": n_fake_total,
        "num_eval": n_eval,
        "real_dataset": "CIFAR10",
        "real_split": "train" if FLAGS.use_train_split else "test",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_path = os.path.join(
        FLAGS.output_dir,
        f"fid_cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Metric summary saved to {out_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app.run(main)
