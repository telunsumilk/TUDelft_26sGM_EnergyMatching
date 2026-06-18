import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


from network_transformer_vit import EBViTModelWrapper

class CIFAR10ModelConfig:
    num_channels = 128
    num_res_blocks = 2
    energy_clamp = None
    num_heads = 4
    num_head_channels = 64
    dropout = 0.1
    attention_resolutions = "16"
    embed_dim = 384
    transformer_nheads = 4
    transformer_nlayers = 8
    output_scale = 1000.0
    channel_mult = [1, 2, 2, 2]


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


def load_cifar_images(folder_path, device, max_images=3):
    """Loads a small subset of CIFAR-10 images for testing."""
    search_pattern = os.path.join(folder_path, "*.[pP][nN][gG]")
    image_paths = glob.glob(search_pattern) + glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]"))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images = []
    for path in image_paths[:max_images]:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        images.append((os.path.basename(path), img_tensor))
    return images


def compute_exact_lid_and_plot(model, image_tensor, filename, device, tau=2.0):
    _, C, H, W = image_tensor.shape
    d = C * H * W
    x_flat = image_tensor.view(-1).clone().detach().requires_grad_(True)

    def energy_fn(x_in):
        t_dummy = torch.zeros(1, device=device)
        v_out = model(t_dummy, x_in.view(1, C, H, W), return_potential=True)
        return v_out.sum()

    print(f"\nProcessing {filename}...")
    print(f"Computing exact Hessian (O(d^3) complexity for d={d}). This will take a few minutes...")

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        hessian_matrix = torch.autograd.functional.hessian(energy_fn, x_flat)

    print("Computing eigenvalue spectrum...")
    eigenvalues, _ = torch.linalg.eigh(hessian_matrix)
    eigenvalues = eigenvalues.detach().cpu().numpy()
    eigenvalues = np.sort(eigenvalues)[::-1]

    near_zero_mask = np.abs(eigenvalues) <= tau
    k = np.sum(near_zero_mask)
    lid = d - k

    print(f"Near-zero eigenvalues (k): {k}")
    print(f"Estimated LID: {lid}")
    print(f"Max eigenvalue: {eigenvalues[0]:.4f}")

    # Plotting
    plt.figure(figsize=(6, 4))
    indices = np.arange(len(eigenvalues))

    plt.plot(indices, eigenvalues, color='blue', linewidth=2, label='Eigenvalues')

    near_zero_indices = indices[near_zero_mask]
    if len(near_zero_indices) > 0:
        plt.axvspan(
            near_zero_indices[0],
            near_zero_indices[-1],
            color='red',
            alpha=0.2,
            label=f'Flat directions ($\\leq {tau}$)'
        )
        plt.scatter(near_zero_indices, eigenvalues[near_zero_mask], color='red', s=10, zorder=3)

    plt.title(f"Exact Hessian spectrum: {filename}")
    plt.suptitle(f"Exact LID: {lid}", x=0.15, y=0.95, fontweight='bold', ha='left')
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.xlim(0, len(eigenvalues))
    plt.text(0.02, 0.05, f'$\\tau = {tau}$', transform=plt.gca().transAxes, fontsize=12, color='gray')

    plt.tight_layout()
    plt.show()

    return lid, eigenvalues


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CHECKPOINT_PATH = "cifar10_checkpoint_final.pt"
    IMAGES_FOLDER = "./cifar10_figures"

    config = CIFAR10ModelConfig()
    model = build_model(config, device)
    model = load_checkpoint(model, CHECKPOINT_PATH, device, use_ema=True)

    # Limit to 3 images
    images_to_process = load_cifar_images(IMAGES_FOLDER, device, max_images=3)

    if not images_to_process:
        print("No images found in the specified folder.")

    for filename, image_tensor in images_to_process:
        # tau=2.0 for CIFAR-10
        compute_exact_lid_and_plot(model, image_tensor, filename, device, tau=2.0)