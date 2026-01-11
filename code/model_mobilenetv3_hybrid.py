
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

# ============================================================
# 4. LOSS FUNCTION
# ============================================================


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - focuses on hard examples (high GT values)
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha  # Focal weight for GT
        self.beta = beta    # Weight for peak regions

    def forward(self, pred, target):
        # Base MSE
        mse = (pred - target) ** 2

        # Focal weight: higher weight where GT is high (near peaks)
        focal_weight = torch.pow(target + 1e-4, self.alpha)

        # Additional penalty for missing peaks
        peak_mask = (target > 0.5).float()
        peak_penalty = peak_mask * self.beta

        # Weighted loss
        weighted_mse = mse * (1 + focal_weight + peak_penalty)

        return weighted_mse.mean()

# ============================================================
# 6. OPTIMIZED INFERENCE FUNCTIONS
# ============================================================

def prepare_for_inference(model):
    """Optimize model for CPU inference"""
    model.eval()

    # Freeze BatchNorm for faster inference
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = False

    # Set num_threads for CPU
    torch.set_num_threads(4)

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    return model


def preprocess_fast(image, img_size=(640, 480)):
    """Optimized preprocessing - faster than original"""
    # Read and convert
    orig_h, orig_w = image.shape[:2]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize directly on numpy (faster)
    img = img.astype(np.float32) * (1.0/255.0)
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
        np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Convert to tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    return img_tensor, orig_h, orig_w


def predict_multi_pallet(model, image_path, device, img_size=(640, 480), stride=4,
                         detection_threshold=0.3, min_distance=5):
    """
    Optimized inference on a single image for multiple pallets
    Returns list of detected center points
    """
    model.eval()

    # Fast preprocessing
    img_tensor, orig_h, orig_w = preprocess_fast(image_path, img_size)
    img_tensor = img_tensor.to(device)

    # Predict with inference_mode (faster than no_grad)
    with torch.inference_mode():
        heatmap = model(img_tensor)

    # Detect multiple peaks - fast version
    hm = heatmap[0, 0].cpu().numpy()
    peaks = detect_peaks(hm, threshold=detection_threshold,
                         min_distance=min_distance)

    # Scale back to original image size
    detections = []
    for x_hm, y_hm, confidence in peaks:
        x_orig = round(x_hm * stride * orig_w / img_size[0])
        y_orig = round(y_hm * stride * orig_h / img_size[1])

        detections.append({
            'x': x_orig,
            'y': y_orig,
            'confidence': confidence
        })

    return detections


def predict_multi_pallet_onnx(session, image, img_size=(640, 480), stride=4,
                              detection_threshold=0.3, min_distance=5):

    # Fast preprocessing
    img_tensor, orig_h, orig_w = preprocess_fast(image, img_size)

    # Inference
    heatmap = session.run(None, {'input': img_tensor.numpy()})[0]

    # Detect peaks
    hm = heatmap[0, 0]
    peaks = detect_peaks(hm, threshold=detection_threshold,
                         min_distance=min_distance)

    # Scale back
    detections = []
    for x_hm, y_hm, confidence in peaks:
        x_orig = round(x_hm * stride * orig_w / img_size[0])
        y_orig = round(y_hm * stride * orig_h / img_size[1])
        detections.append({'x': x_orig, 'y': y_orig, 'confidence': confidence})

    return detections


class MobileNetV3Hybrid_Balanced(nn.Module):
    """
    HYBRID VERSION - FIXED - Cân bằng tốc độ và độ chính xác

    CRITICAL FIXES:
    1. Dùng BatchNorm thay GroupNorm (tốt hơn cho heatmap regression)
    2. Giữ channels: 576 → 192 → 96 → 48 (trung bình 2 phiên bản)
    3. CÓ Sigmoid ở output (như model gốc)
    4. Depthwise separable ở refine layer

    Expected performance:
    - Speed: ~80-85% của Fast version
    - Accuracy: ~92-95% của Accurate version

    Input:  (B, 3, 480, 640)
    Output: (B, 1, 120, 160)  [stride=4]
    """

    def __init__(self):
        super().__init__()

        # Backbone: MobileNetV3-Small
        mobilenet = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features
        # Output: (B, 576, 15, 20)

        # Decoder: Gradual channel reduction với BatchNorm
        self.upsample = nn.Sequential(
            # 32 -> 16: (15, 20) -> (30, 40)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(576, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),  # ✅ FIXED: Dùng BatchNorm
            nn.ReLU(inplace=True),

            # 16 -> 8: (30, 40) -> (60, 80)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(192, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),   # ✅ FIXED: Dùng BatchNorm
            nn.ReLU(inplace=True),

            # 8 -> 4: (60, 80) -> (120, 160)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(96, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),   # ✅ FIXED: Dùng BatchNorm
            nn.ReLU(inplace=True),
        )

        # Refinement: Lightweight với standard conv
        self.refine = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Head: WITH Sigmoid (như model gốc)
        self.head = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=1),
            nn.Sigmoid()  # ✅ Output [0,1]
        )

    def forward(self, x):
        x = self.backbone(x)   # (B, 576, 15, 20)
        x = self.upsample(x)   # (B, 48, 120, 160)
        x = self.refine(x)     # (B, 48, 120, 160)
        x = self.head(x)       # (B, 1, 120, 160)
        return x


# ============================================================
# FIXED LOSS FUNCTION
# ============================================================
class FocalMSELoss_Fixed(nn.Module):
    """
    FIXED Focal MSE Loss - Không apply sigmoid nữa vì model đã có sigmoid

    CRITICAL FIX: Model output đã là [0,1], không cần sigmoid lại
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        # ✅ FIXED: pred đã là [0,1], KHÔNG apply sigmoid nữa
        # pred = torch.sigmoid(pred)  # ❌ REMOVED

        # Base MSE
        mse = (pred - target) ** 2

        # Focal weight
        focal_weight = torch.pow(target + 1e-4, self.alpha)

        # Peak penalty
        peak_mask = (target > 0.5).float()
        peak_penalty = peak_mask * self.beta

        # Weighted loss
        weighted_mse = mse * (1 + focal_weight + peak_penalty)

        return weighted_mse.mean()

def load_model_onnx(path_model_onnx):
    session = ort.InferenceSession(
        path_model_onnx,
        providers=['CPUExecutionProvider']
    )
    return session


path_model_onnx = r'C:\Lap trinh\realsense\final_project\model\onnx_hybrid\model_fast.onnx'
path_image = r"C:\Lap trinh\realsense\project\images\rgb\0113.png"


# if __name__ == "__main__":
#     image = cv2.imread(path_image)
#     session = load_model_onnx(path_model_onnx)
#     start = time.time()
#     detections = predict_multi_pallet_onnx(session, image)
#     end = time.time()
#     print(end - start)

#     for detection in detections:
#         cv2.circle(image, (detection['x'], detection['y']), 4, (0, 255, 0), -1)
#         print(detection['confidence'])
#     cv2.imshow('res', image)
    
#     cv2.waitKey(0)
    
if __name__ == "__main__":
    # export_to_onnx(path_model_center, output_path='./new_project/resnet18_multi_center.onnx')
    list_time = []
    pipeline, align = get_image.pre()
    session = load_model_onnx(path_model_onnx)
    while True:
        rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
        start = time.time()
        detections = predict_multi_pallet_onnx(session, rgb)
        end = time.time()
        list_time.append(end - start)
        print("confidence:", detections[0]['confidence'] if len(
            detections) > 0 else "No detections")
        for detection in detections:
            if detection['confidence'] > 0.2:
                cv2.circle(
                    rgb, (detection['x'], detection['y']), 4, (0, 255, 0), -1)
        cv2.imshow('res', rgb)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print("Average inference time:", sum(list_time[1:])/len(list_time))
            cv2.destroyAllWindows()
            break
