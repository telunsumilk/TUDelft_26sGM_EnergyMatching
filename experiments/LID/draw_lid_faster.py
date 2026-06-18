import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from network_transformer_vit import EBViTModelWrapper

class ModelConfig:
    num_channels = 32
    num_res_blocks = 2
    energy_clamp = None
    num_heads = 2
    num_head_channels = 64
    dropout = 0.1
    attention_resolutions = "16"  # Fixed from list to string
    embed_dim = 128
    transformer_nheads = 2
    transformer_nlayers = 2
    output_scale = 100.0
    channel_mult = [1, 2, 2]


def build_model(config, device):
    model = EBViTModelWrapper(
        dim=(3, 32, 32),
        num_channels=config.num_channels,
        num_res_blocks=config.num_res_blocks,
        channel_mult=config.channel_mult,
        attention_resolutions=config.attention_resolutions,
        num_heads=config.num_heads,
        num_head_channels=config.num_head_channels,
        dropout=config.dropout,
        output_scale=config.output_scale,
        energy_clamp=config.energy_clamp,
        patch_size=4,
        embed_dim=config.embed_dim,
        transformer_nheads=config.transformer_nheads,
        transformer_nlayers=config.transformer_nlayers,
    ).to(device)
    return model.eval()


def load_checkpoint(model, ckpt_path, device, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location=device)
    key = "ema_model" if use_ema else "net_model"
    model.load_state_dict(ckpt[key], strict=True)
    print(f"Loaded {key} from {ckpt_path}")
    return model


def load_images_from_folder(folder_path, device):
    search_pattern = os.path.join(folder_path, "*.[pP][nN][gG]")
    image_paths = glob.glob(search_pattern) + glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]"))

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        images.append((os.path.basename(path), img_tensor))
    return images


def randomized_hessian_spectrum(energy_fn, x_flat, d, m=256):
    device = x_flat.device
    print(f"  -> Projecting to {m}-dimensional subspace...")

    Omega = torch.randn(d, m, device=device)
    Y = torch.zeros(d, m, device=device)

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for i in range(m):
            v = Omega[:, i]
            _, hvp_out = torch.autograd.functional.hvp(energy_fn, x_flat, v)
            Y[:, i] = hvp_out

    Q, _ = torch.linalg.qr(Y)

    W = torch.zeros(d, m, device=device)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for i in range(m):
            v = Q[:, i]
            _, hvp_out = torch.autograd.functional.hvp(energy_fn, x_flat, v)
            W[:, i] = hvp_out

    B = Q.t() @ W

    eigenvalues, _ = torch.linalg.eigh(B)

    eigenvalues = eigenvalues.detach().cpu().numpy()
    return np.sort(eigenvalues)[::-1]


def compute_lid_fast_and_plot(model, image_tensor, filename, device, tau=3.0, m=512):
    _, C, H, W = image_tensor.shape
    d = C * H * W
    x_flat = image_tensor.view(-1).clone().detach().requires_grad_(True)

    def energy_fn(x_in):
        t_dummy = torch.zeros(1, device=device)
        v_out = model(t_dummy, x_in.view(1, C, H, W), return_potential=True)
        return v_out.sum()

    print(f"\nProcessing {filename} (Fast Method)...")

    top_eigenvalues = randomized_hessian_spectrum(energy_fn, x_flat, d, m=m)
    lid = np.sum(np.abs(top_eigenvalues) > tau)

    print(f"Estimated LID: {lid}")
    print(f"Max eigenvalue: {top_eigenvalues[0]:.4f}")

    plt.figure(figsize=(6, 4))

    full_spectrum = np.zeros(d)
    full_spectrum[:m] = top_eigenvalues
    indices = np.arange(d)

    plt.plot(indices, full_spectrum, color='blue', linewidth=2, label='Eigenvalues')

    near_zero_mask = np.abs(full_spectrum) <= tau
    near_zero_indices = indices[near_zero_mask]

    if len(near_zero_indices) > 0:
        plt.axvspan(
            near_zero_indices[0],
            near_zero_indices[-1],
            color='red',
            alpha=0.2,
            label=f'Flat directions ($\\leq {tau}$)'
        )
        plt.scatter(near_zero_indices, full_spectrum[near_zero_mask], color='red', s=5, zorder=3)

    plt.title(f"Hessian spectrum: {filename} (Rank-{m} Approx)")
    plt.suptitle(f"LID: {lid}", x=0.15, y=0.95, fontweight='bold', ha='left')
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.xlim(0, d)
    plt.text(0.02, 0.05, f'$\\tau = {tau}$', transform=plt.gca().transAxes, fontsize=12, color='gray')

    plt.tight_layout()
    plt.show()

    return lid, full_spectrum


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CHECKPOINT_PATH = "checkpoint_50400.pt"
    IMAGES_FOLDER = "./mnist_figures"

    config = ModelConfig()
    model = build_model(config, device)
    model = load_checkpoint(model, CHECKPOINT_PATH, device, use_ema=True)

    images_to_process = load_images_from_folder(IMAGES_FOLDER, device)

    for filename, image_tensor in images_to_process:
        # increase M incase lid all equal
        compute_lid_fast_and_plot(model, image_tensor, filename, device, tau=3.0, m=1024)