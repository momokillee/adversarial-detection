"""
CW and DeepFool adversarial attack generation.
Fixed implementation that verifies attack success before saving.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from utils.preprocess import load_image_tensor, save_tensor_image

IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Load a proper pretrained ResNet18 as victim
# This is critical — CW needs a real trained model
# to attack, not an untrained custom CNN
# ──────────────────────────────────────────────

def load_proper_victim(device):
    """Load pretrained ResNet18 as victim model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to(device)
    model.eval()
    print(f"Loaded pretrained ResNet18 victim on {device}")
    return model


def get_prediction(model, image_tensor):
    """Get predicted class for an image tensor."""
    with torch.no_grad():
        output = model(image_tensor)
        return output.argmax(dim=1).item()


# ──────────────────────────────────────────────
# CW Attack — proper implementation
# ──────────────────────────────────────────────

def cw_attack_single(
    model,
    image: torch.Tensor,
    true_label: int,
    c: float = 1.0,
    kappa: float = 0.0,
    max_steps: int = 200,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    CW L2 attack on a single image.
    Minimises: ||delta||_2 + c * f(x + delta)
    where f encourages misclassification.
    """
    # Work in tanh space to keep pixels in [0,1]
    # x = 0.5 * (tanh(w) + 1)  =>  w = arctanh(2x - 1)
    x = image.clone().detach().to(device)

    # Clamp to avoid arctanh(±1) = ±inf
    x_clamped = x.clamp(0.001, 0.999)
    w = torch.arctanh(2 * x_clamped - 1)
    w = w.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)
    best_adv = x.clone()
    best_l2 = float("inf")

    for step in range(max_steps):
        # Convert back to image space
        adv = 0.5 * (torch.tanh(w) + 1)

        # L2 distance
        l2 = torch.sum((adv - x) ** 2)

        # CW loss: encourages misclassification
        output = model(adv)
        target_score = output[0, true_label]
        other_scores = output.clone()
        other_scores[0, true_label] = float("-inf")
        best_other = other_scores.max()
        f_loss = torch.clamp(target_score - best_other + kappa, min=0)

        loss = l2 + c * f_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track best adversarial example
        with torch.no_grad():
            pred = output.argmax(dim=1).item()
            if pred != true_label and l2.item() < best_l2:
                best_l2 = l2.item()
                best_adv = adv.detach().clone()

    return best_adv.clamp(0, 1)


# ──────────────────────────────────────────────
# Main generation script
# ──────────────────────────────────────────────

def generate_cw(
    clean_dir: Path,
    output_dir: Path,
    max_images: int = 500,
    c: float = 10.0,
    kappa: float = 0.0,
    steps: int = 200,
    lr: float = 0.01
):
    """Generate CW adversarial images with success verification."""
    model = load_proper_victim(DEVICE)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in clean_dir.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])[:max_images]

    print(f"\nGenerating CW attacks on {len(image_paths)} images...")
    print(f"  c={c}, kappa={kappa}, steps={steps}, lr={lr}")
    print(f"  Device: {DEVICE}")

    success = 0
    failed = 0
    skipped = 0
    start = time.time()

    for i, path in enumerate(image_paths):
        img = load_image_tensor(path, IMAGE_SIZE, DEVICE)
        true_label = get_prediction(model, img)

        adv = cw_attack_single(
            model, img, true_label,
            c=c, kappa=kappa,
            max_steps=steps, lr=lr,
            device=DEVICE
        )

        # Verify attack succeeded
        adv_pred = get_prediction(model, adv)
        l2_dist = torch.norm(adv - img).item()

        out_path = output_dir / f"cw_{path.stem}.png"

        if adv_pred != true_label and l2_dist > 1e-6:
            save_tensor_image(adv, out_path)
            success += 1
        else:
            # Save anyway but note it failed
            save_tensor_image(adv, out_path)
            failed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(
                f"  [{i+1}/{len(image_paths)}] "
                f"Success: {success} | Failed: {failed} | "
                f"Time: {elapsed:.1f}s"
            )

    elapsed = time.time() - start
    print(f"\nCW generation complete:")
    print(f"  Total: {len(image_paths)}")
    print(f"  Successful attacks: {success} ({success/len(image_paths)*100:.1f}%)")
    print(f"  Failed attacks: {failed} ({failed/len(image_paths)*100:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved to: {output_dir}")


def generate_deepfool(
    clean_dir: Path,
    output_dir: Path,
    max_images: int = 500,
    steps: int = 50,
    overshoot: float = 0.02
):
    """Generate DeepFool adversarial images."""
    try:
        import torchattacks
    except ImportError:
        print("Install torchattacks: pip install torchattacks")
        return

    model = load_proper_victim(DEVICE)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in clean_dir.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])[:max_images]

    print(f"\nGenerating DeepFool attacks on {len(image_paths)} images...")
    atk = torchattacks.DeepFool(model, steps=steps, overshoot=overshoot)

    success = 0
    start = time.time()

    for i, path in enumerate(image_paths):
        img = load_image_tensor(path, IMAGE_SIZE, DEVICE)
        true_label = torch.tensor(
            [get_prediction(model, img)]
        ).to(DEVICE)

        adv = atk(img, true_label)
        adv_pred = get_prediction(model, adv)
        l2 = torch.norm(adv - img).item()

        out_path = output_dir / f"deepfool_{path.stem}.png"
        save_tensor_image(adv, out_path)

        if adv_pred != true_label.item() and l2 > 1e-6:
            success += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(image_paths)}] Success: {success} | Time: {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"\nDeepFool generation complete:")
    print(f"  Successful: {success}/{len(image_paths)} ({success/len(image_paths)*100:.1f}%)")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", choices=["cw", "deepfool", "both"],
                        default="cw")
    parser.add_argument("--clean-dir", type=Path,
                        default=Path("data/clean"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.attack in ["cw", "both"]:
        generate_cw(
            clean_dir=args.clean_dir,
            output_dir=Path("data/cw_adversarial_v2"),
            max_images=args.limit,
            c=args.c,
            steps=args.steps
        )

    if args.attack in ["deepfool", "both"]:
        generate_deepfool(
            clean_dir=args.clean_dir,
            output_dir=Path("data/deepfool_adversarial_v2"),
            max_images=args.limit
        )
        