import onnxruntime as ort
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import time
import get_image
import os
from scipy.ndimage import maximum_filter

# ============================================================
class MobileNetV3MultiPalletHeatmap(nn.Module):
    """
    Ultra-optimized MobileNetV3-Small for CPU/OpenVINO
    Reduced channels: 576 → 128 → 64 → 32 → 1
    NO Sigmoid in model (for better quantization)
    Input:  (B, 3, 480, 640)
    Output: (B, 1, 120, 160)  [stride=4]
    """

    def __init__(self):
        super().__init__()

        # Load pretrained MobileNetV3-Small
        mobilenet = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features
        # Output: (B, 576, H/32, W/32) = (B, 576, 15, 20)

        # Ultra-lightweight decoder - OPTIMIZED CHANNELS
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(576, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Lightweight refinement - REDUCED
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Head: NO SIGMOID (better for quantization)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 480, 640)
        x = self.backbone(x)      # (B, 576, 15, 20)
        x = self.upsample(x)       # (B, 32, 120, 160)
        x = self.refine(x)         # (B, 32, 120, 160)
        x = self.head(x)           # (B, 1, 120, 160) - NO SIGMOID!
        return x


# # Alias for compatibility
# detect_peaks = detect_peaks_opencv
# detect_peaks_fast = detect_peaks_opencv


# ============================================================
# 4. LOSS FUNCTION - ADAPTED FOR NO SIGMOID
# ============================================================

class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - adapted for model without sigmoid
    Applies sigmoid internally for training
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)

        mse = (pred_sig - target) ** 2

        # focal on prediction confidence
        focal_weight = torch.pow(1.0 - pred_sig, self.alpha)

        peak_penalty = (target > 0.5).float() * self.beta

        loss = mse * (1 + focal_weight + peak_penalty)
        return loss.mean()

# ============================================================
# 6. OPTIMIZED INFERENCE
# ============================================================

def prepare_for_inference(model, num_threads=None):
    """Optimize model for CPU inference"""
    model.eval()

    # Freeze BatchNorm
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = False

    # Set num_threads dynamically
    if num_threads is None:
        num_threads = max(1, os.cpu_count() // 2)
    torch.set_num_threads(num_threads)
    print(f"✓ Set num_threads to {num_threads}")

    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False

    return model


def preprocess_fast(image, img_size=(640, 480)):
    """Optimized preprocessing"""
    
    orig_h, orig_w = image.shape[:2]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)

    # Normalize directly on numpy
    img = img.astype(np.float32) * (1.0/255.0)
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
        np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    return img_tensor, orig_h, orig_w


def predict_multi_pallet(model, image_path, device, img_size=(640, 480), stride=4,
                         detection_threshold=0.5, min_distance=5):
    """
    Optimized inference with sigmoid applied externally
    """
    model.eval()

    img_tensor, orig_h, orig_w = preprocess_fast(image_path, img_size)
    img_tensor = img_tensor.to(device)

    with torch.inference_mode():
        heatmap_raw = model(img_tensor)

        # Apply sigmoid externally (model has no sigmoid)
        heatmap = torch.sigmoid(heatmap_raw)

    # Detect peaks
    hm = heatmap[0, 0].cpu().numpy()
    peaks = detect_peaks(
        hm, threshold=detection_threshold, min_distance=min_distance)

    # Scale back
    detections = []
    for x_hm, y_hm, confidence in peaks:
        x_orig = round(x_hm * stride * orig_w / img_size[0])
        y_orig = round(y_hm * stride * orig_h / img_size[1])
        detections.append({'x': x_orig, 'y': y_orig, 'confidence': confidence})

    return detections


def detect_peaks(heatmap, threshold=0.3, min_distance=5):
    """
    Detect multiple peaks in heatmap using non-maximum suppression

    Args:
        heatmap: (H, W) numpy array
        threshold: minimum confidence threshold
        min_distance: minimum distance between peaks (in heatmap pixels)

    Returns:
        List of (x, y, confidence) tuples
    """
    # Apply threshold
    heatmap_thresh = heatmap * (heatmap > threshold)

    # Non-maximum suppression using maximum filter
    footprint = np.ones((min_distance*2+1, min_distance*2+1))
    local_max = maximum_filter(
        heatmap_thresh, footprint=footprint) == heatmap_thresh

    # Get peak locations
    peaks = np.argwhere(local_max & (heatmap_thresh > 0))

    # Extract confidences and sort by confidence
    detections = []
    for y, x in peaks:
        confidence = heatmap[y, x]
        detections.append((int(x), int(y), float(confidence)))

    # Sort by confidence descending
    detections.sort(key=lambda d: d[2], reverse=True)

    return detections

def predict_multi_pallet_onnx(session, image, img_size=(640, 480), stride=4,
                              detection_threshold=0.3, min_distance=5):

    img_tensor, orig_h, orig_w = preprocess_fast(image, img_size)

    # Inference
    heatmap_raw = session.run(None, {'input': img_tensor.numpy()})[0]

    # Apply sigmoid (model has no sigmoid)
    heatmap = 1 / (1 + np.exp(-heatmap_raw))

    # Detect peaks
    hm = heatmap[0, 0]
    peaks = detect_peaks(
        hm, threshold=detection_threshold, min_distance=min_distance)

    detections = []
    for x_hm, y_hm, confidence in peaks:
        x_orig = round(x_hm * stride * orig_w / img_size[0])
        y_orig = round(y_hm * stride * orig_h / img_size[1])
        detections.append({'x': x_orig, 'y': y_orig, 'confidence': confidence})

    return detections

path_model_center = r"C:\Lap trinh\realsense\final_project\model\best_model.pth"
path_model_onnx = r"C:\Lap trinh\realsense\final_project\model\model_onnx.onnx"

if __name__ == "__main__":
            list_time = []
            device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")

            print(f"Using device: {device}")
            # --- Tải Mô hình ---
            model = MobileNetV3MultiPalletHeatmap().to(device)

            if os.path.exists(path_model_center):
                checkpoint = torch.load(path_model_center, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.6f})")
            else:
                print(
                    f"Warning: Checkpoint not found at {path_model_center}. Using uninitialized model.")

            model.eval()
            pipeline, align = get_image.pre()
            while True:
                rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
                start = time.time()
                detections = predict_multi_pallet(model, rgb, device)
                end = time.time()
                list_time.append(end - start)
                print("confidence:", detections[0]['confidence'] if len(detections) > 0 else "No detections")
                for detection in detections:
                    if detection['confidence'] > 0.5:
                        cv2.circle(rgb, (detection['x'], detection['y']), 4, (0, 255, 0), -1)
                cv2.imshow('res', rgb)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    print("Average inference time:", sum(list_time)/len(list_time))
                    cv2.destroyAllWindows()
                    break

def load_model_onnx(path_model_onnx):
    session = ort.InferenceSession(
        path_model_onnx,
        providers=['CPUExecutionProvider']
    )
    return session

if __name__ == "__main__":
    # export_to_onnx(path_model_center, output_path='./new_project/resnet18_multi_center.onnx')
    list_time = []
    session = load_model_onnx(path_model_onnx)
    pipeline, align = get_image.pre()
    while True:
        rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
        start = time.time()
        detections = predict_multi_pallet_onnx(
            session, rgb, detection_threshold=0.3)
        end = time.time()
        print(end - start)
        print("confidence:", detections[0]['confidence'] if len(
            detections) > 0 else "No detections")
        list_time.append(end - start)
        for detection in detections:
            if detection['confidence'] > 0.2:
                cv2.circle(
                    rgb, (detection['x'], detection['y']), 4, (0, 255, 0), -1)
        cv2.imshow('res', rgb)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print("Average inference time:", sum(list_time)/len(list_time))
            cv2.destroyAllWindows()
            break


# folder = r"C:\Lap trinh\realsense\image"
# path_save = r"C:\Lap trinh\realsense\output\New folder (2)"
# if __name__ == "__main__":
#     session = load_model_onnx(path_model_onnx)
#     files = []
#     for f in os.listdir(folder):
#         if f.lower().endswith(".png"):
#             files.append(os.path.join(folder, f))

#     for file in files:
#         image = cv2.imread(file)
#         detections = predict_multi_pallet_onnx(
#             session, image, detection_threshold=0.3)
#         for detection in detections:
#             cv2.circle(image, (detection['x'],
#                                detection['y']), 4, (0, 255, 0), -1)
#             cv2.putText(image, f"{detection['confidence']:.2f}", (detection['x'], detection['y']-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         cv2.imshow('res', image)
#         cv2.imwrite(os.path.join(path_save, os.path.basename(file)), image)
#         cv2.waitKey(0)
