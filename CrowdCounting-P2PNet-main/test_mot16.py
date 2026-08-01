"""
اسکریپت تست مدل تخصصی P2PNet روی MOT16 و TownCentreXVID
باید داخل پوشه‌ی ریپوی CrowdCounting-P2PNet اجرا بشه، چون به پکیج models نیاز داره.
"""

import os
import glob
import time
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

from models import build_model  # این از خود ریپوی P2PNet میاد


# ==========================================
# 1. آرگومان‌های لازم برای ساخت مدل
# (دقیقاً همون مقادیری که در آموزش/دمو رسمی استفاده شده)
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth', type=str)
    parser.add_argument('--image_dir', default='MOT16-03/img1', type=str,
                         help='پوشه‌ی فریم‌های MOT16 (یا هر پوشه عکس دیگه)')
    parser.add_argument('--video_path', default='', type=str,
                         help='اگه پر باشه، به‌جای پوشه عکس، ویدیو پردازش میشه (مثلاً TownCentreXVID.mp4)')
    parser.add_argument('--output_video', default='output_p2pnet.mp4', type=str)
    parser.add_argument('--threshold', default=0.5, type=float,
                         help='آستانه‌ی اطمینان برای قبول یک نقطه به‌عنوان یک نفر')
    return parser.parse_args()


def load_model(args, device):
    model = build_model(args)
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model


transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
])


# ==========================================
# 2. اجرای مدل روی یک فریم (numpy BGR از cv2)
# ==========================================
def run_on_frame(model, frame_bgr, device, threshold):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # نکته مهم: P2PNet نیاز داره سایز ورودی مضربی از 128 باشه
    width, height = pil_img.size
    new_width = max(128, (width // 128) * 128)
    new_height = max(128, (height // 128) * 128)
    pil_resized = pil_img.resize((new_width, new_height), Image.LANCZOS)

    img_tensor = transform(pil_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]

    mask = outputs_scores > threshold
    points = outputs_points[mask].detach().cpu().numpy()  # نقاط به مقیاس new_width/new_height
    count = int(mask.sum())

    # نگاشت نقاط به سایز اصلی فریم (چون resize کردیم)
    scale_x = frame_bgr.shape[1] / new_width
    scale_y = frame_bgr.shape[0] / new_height
    points_orig = points.copy()
    if len(points_orig) > 0:
        points_orig[:, 0] *= scale_x
        points_orig[:, 1] *= scale_y

    return count, points_orig


# ==========================================
# 3. ساخت هیت‌مپ از نقاط (برای هم‌خونی با گزارش قبلی که با CSRNet هیت‌مپ داشتی)
# ==========================================
def make_heatmap(frame_shape, points, sigma=25):
    h, w = frame_shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (x, y) in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            heat[yi, xi] += 1.0
    if len(points) > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    return heat_color, heat_norm


def overlay_heatmap(frame, heat_color, heat_norm, alpha_max=0.6):
    alpha = (heat_norm / 255.0) * alpha_max
    alpha = np.expand_dims(alpha, axis=-1)
    return (frame * (1 - alpha) + heat_color * alpha).astype(np.uint8)


# ==========================================
# 4. پردازش یک پوشه عکس (برای MOT16)
# ==========================================
def process_image_folder(model, args, device):
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not image_paths:
        print(f"هیچ عکسی توی {args.image_dir} پیدا نشد.")
        return

    first = cv2.imread(image_paths[0])
    h, w = first.shape[:2]
    out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h))

    total_counts = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        t0 = time.time()
        count, points = run_on_frame(model, frame, device, args.threshold)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        total_counts.append(count)

        heat_color, heat_norm = make_heatmap(frame.shape, points)
        final_frame = overlay_heatmap(frame, heat_color, heat_norm)

        for (x, y) in points:
            cv2.circle(final_frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        cv2.rectangle(final_frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.putText(final_frame, f"Count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(final_frame, f"FPS: {fps:.1f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(final_frame)
        print(f"{os.path.basename(img_path)} -> count={count}")

        cv2.imshow("P2PNet - MOT16", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    out.release()
    print(f"\nمیانگین شمارش روی این دنباله: {np.mean(total_counts):.2f}")
    print(f"ویدیوی خروجی ذخیره شد: {args.output_video}")


# ==========================================
# 5. پردازش یک ویدیو (برای TownCentreXVID.mp4)
# ==========================================
def process_video(model, args, device):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"باز کردن ویدیو {args.video_path} ممکن نشد.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h))

    total_counts = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        count, points = run_on_frame(model, frame, device, args.threshold)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        total_counts.append(count)

        heat_color, heat_norm = make_heatmap(frame.shape, points)
        final_frame = overlay_heatmap(frame, heat_color, heat_norm)
        for (x, y) in points:
            cv2.circle(final_frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        cv2.rectangle(final_frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.putText(final_frame, f"Count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(final_frame, f"FPS: {fps:.1f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(final_frame)
        if frame_idx % 30 == 0:
            print(f"frame {frame_idx} -> count={count}")
        frame_idx += 1

        cv2.imshow("P2PNet - TownCentre", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()
    print(f"\nمیانگین شمارش روی این ویدیو: {np.mean(total_counts):.2f}")
    print(f"ویدیوی خروجی ذخیره شد: {args.output_video}")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args, device)

    if args.video_path:
        process_video(model, args, device)
    else:
        process_image_folder(model, args, device)


if __name__ == "__main__":
    main()
