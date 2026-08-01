import os
import glob
import time

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models import vgg19

# =====================================================
# Configuration
# =====================================================

MODEL_PATH = "pretrained_models/model_qnrf.pth"
IMAGE_DIR = "/home/mohamad/Desktop/Embedded_Systems/project/MOT16-03/img1"
OUTPUT_VIDEO = "dmcount_output.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA = 0.55

# =====================================================
# Load Model
# =====================================================

model = vgg19()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)

model.eval()

transform = transforms.ToTensor()

# =====================================================

image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

if len(image_paths) == 0:
    raise RuntimeError("No images found.")

first = cv2.imread(image_paths[0])

H, W = first.shape[:2]

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (W, H)
)

display_count = 0
COUNT_SMOOTHING = 0.20

print(f"Processing {len(image_paths)} frames...")

# =====================================================

for img_path in image_paths:

    frame = cv2.imread(img_path)

    if frame is None:
        continue

    start = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(rgb)

    inp = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        density, _ = model(inp)

    ####################################################
    # Count
    ####################################################

    person_count = density.sum().item()

    display_count = (
        (1 - COUNT_SMOOTHING) * display_count
        + COUNT_SMOOTHING * person_count
    )

    ####################################################
    # Density Map
    ####################################################

    heatmap = density[0, 0].cpu().numpy()

    heatmap = (
        heatmap - heatmap.min()
    ) / (
        heatmap.max() - heatmap.min() + 1e-5
    )

    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap = cv2.resize(
        heatmap,
        (W, H),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = cv2.GaussianBlur(
        heatmap,
        (51, 51),
        0
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    ####################################################
    # Dynamic Alpha
    ####################################################

    alpha = (
        heatmap.astype(np.float32) / 255.0
    ) * ALPHA

    alpha = alpha[..., None]

    final = (
        frame * (1 - alpha)
        + heatmap_color * alpha
    ).astype(np.uint8)

    ####################################################
    # FPS
    ####################################################

    fps = 1 / (time.time() - start)

    cv2.rectangle(
        final,
        (10, 10),
        (340, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        final,
        f"Count: {display_count:.1f}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        final,
        f"FPS: {fps:.1f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    writer.write(final)

    cv2.imshow("DM-Count", final)

    if cv2.waitKey(1) == ord("q"):
        break

writer.release()

cv2.destroyAllWindows()

print("Saved:", OUTPUT_VIDEO)
