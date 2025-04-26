from pathlib import Path  # Platform-independent file paths
from typing import List  # Type hinting for lists

import cv2  # OpenCV for image processing
import torch  # PyTorch core library
import torchvision.transforms as T  # Image transformations
from pytorch_grad_cam import GradCAM  # Grad-CAM visualization tool
from pytorch_grad_cam.utils.image import show_cam_on_image  # Utility to overlay CAM on images
from torchvision import models  # Predefined models like VGG19

# ---------------------------------------------------------------------------
# CONFIGURATION - Editable parameters
# ---------------------------------------------------------------------------
WEIGHTS_PATH = Path("real_fake_classifier.pth")  # Path to the trained model weights
INPUT_DIR    = Path("test")  # Folder containing the input images
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
RECURSIVE    = True  # Whether to search inside subdirectories
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def last_conv(module: torch.nn.Module):
    """Return the last Conv2d layer found in the module."""
    for layer in reversed(list(module.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return [layer]
    raise RuntimeError("No Conv2d layer found; specify target_layers manually.")

def find_images(root: Path) -> List[Path]:
    """Find and sort all supported image files under a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in root.rglob("*" if RECURSIVE else "*") if p.suffix.lower() in exts]
    return sorted(images)

# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------

def main():
    # Initialize model and device
    device = DEVICE

    # 1. Create VGG19 model architecture (untrained)
    model = models.vgg19(weights=None)
    model.classifier[6] = torch.nn.Linear(4096, 2)  # Adjust the last layer for 2 classes

    # 2. Load pretrained weights
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # 3. Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Identify the last convolutional layer for Grad-CAM
    target_layers = last_conv(model)

    # Define the preprocessing pipeline
    preprocess = T.Compose([
        T.ToTensor(),
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Find input images
    images = find_images(INPUT_DIR)
    if not images:
        raise FileNotFoundError(f"No supported image files found under {INPUT_DIR}")

    print(f"[i] Found {len(images)} images. Processing…")

    # Visualization with Grad-CAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for idx, img_path in enumerate(images, 1):
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[!] Skipping unreadable file: {img_path}")
                continue

            rgb_uint8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_float = rgb_uint8.astype("float32") / 255.0

            tensor = preprocess(rgb_uint8).unsqueeze(0).to(device)

            cam_map = cam(input_tensor=tensor)[0]
            cam_map = cv2.resize(cam_map, (rgb_float.shape[1], rgb_float.shape[0]))

            overlay = show_cam_on_image(rgb_float, cam_map, use_rgb=True)

            # Show the image
            window_name = f"Grad-CAM: {img_path.name}"
            cv2.imshow(window_name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)
            cv2.destroyWindow(window_name)

            if key == 27:  # ESC key pressed
                print("[x] ESC pressed. Exiting early.")
                return

            print(f"[{idx}/{len(images)}] Displayed → {img_path.name}")

    print("[✓] All done!")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
