
import os
"""Fast inference using ONNX Runtime"""
import onnxruntime as ort
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import maximum_filter, label, center_of_mass
import time
import get_image
path_model_center = r"C:\Users\Lenovo\Downloads\best_model_mobilenetv3_2.pth"
model_onnx_path = r"C:\Users\Lenovo\Downloads\data\model_fast.onnx"

class MobileNetV3MultiPalletHeatmap(nn.Module):
    """
    MobileNetV3-Small backbone with optimized decoder
    ~2-3x faster than ResNet18 on CPU
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

        # Lightweight decoder: Upsample + Conv (faster than ConvTranspose2d)
        self.upsample = nn.Sequential(
            # 32 -> 16: (15, 20) -> (30, 40)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(576, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 16 -> 8: (30, 40) -> (60, 80)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 8 -> 4: (60, 80) -> (120, 160)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Refinement layer - reduced to 1 layer
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Head: project to single center heatmap
        self.head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 3, 480, 640)
        x = self.backbone(x)      # (B, 576, 15, 20)
        x = self.upsample(x)       # (B, 64, 120, 160)
        x = self.refine(x)
        x = self.head(x)           # (B, 1, 120, 160)
        return x


# ============================================================
# 3. OPTIMIZED PEAK DETECTION
# ============================================================

def simple_nms(peaks, min_distance):
    """Fast non-maximum suppression"""
    keep = [peaks[0]]

    for peak in peaks[1:]:
        x, y, conf = peak

        # Check distance to kept peaks
        far_enough = all(
            (x - kx)**2 + (y - ky)**2 >= min_distance**2
            for kx, ky, _ in keep
        )

        if far_enough:
            keep.append(peak)

    return keep


def preprocess_fast(image, img_size=(640, 480)):
    """Optimized preprocessing - faster than original"""
    # Read and convert
    
    orig_h, orig_w = image.shape[:2]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)

    # Normalize directly on numpy (faster)
    img = img.astype(np.float32) * (1.0/255.0)
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
        np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Convert to tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    return img_tensor, orig_h, orig_w


def predict_multi_pallet(model, image, device, img_size=(640, 480), stride=4,
                         detection_threshold=0.3, min_distance=5):
    """
    Optimized inference on a single image for multiple pallets
    Returns list of detected center points
    """
    # Fast preprocessing
    img_tensor, orig_h, orig_w = preprocess_fast(image, img_size)
    img_tensor = img_tensor.to(device)

    # Predict with inference_mode (faster than no_grad)
    with torch.inference_mode():
        heatmap = model(img_tensor)

    # Detect multiple peaks - fast version
    hm = heatmap[0, 0].cpu().numpy()
    peaks = detect_peaks(
        hm, threshold=detection_threshold, min_distance=min_distance)

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

def export_to_onnx(checkpoint_path, output_path='model_mobilenet.onnx'):
    """Export MobileNetV3 model to ONNX format"""
    device = torch.device('cpu')

    # Load model
    model = MobileNetV3MultiPalletHeatmap()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = prepare_for_inference(model)

    # Dummy input
    dummy_input = torch.randn(1, 3, 480, 640)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )
    print(f"✓ Model exported to {output_path}")


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

def predict_multi_pallet_onnx(session, image_path, img_size=(640, 480), stride=4,
                              detection_threshold=0.3, min_distance=5):

    # Fast preprocessing
    img_tensor, orig_h, orig_w = preprocess_fast(image_path, img_size)

    # Inference
    heatmap = session.run(None, {'input': img_tensor.numpy()})[0]

    # Detect peaks
    hm = heatmap[0, 0]
    peaks = detect_peaks(
        hm, threshold=detection_threshold, min_distance=min_distance)

    # Scale back
    detections = []
    for x_hm, y_hm, confidence in peaks:
        x_orig = round(x_hm * stride * orig_w / img_size[0])
        y_orig = round(y_hm * stride * orig_h / img_size[1])
        detections.append({'x': x_orig, 'y': y_orig, 'confidence': confidence})

    return detections

# if __name__ == "__main__":
#         list_time = []
#         device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
#         # device = torch.device("cpu")

#         print(f"Using device: {device}")
#         # --- Tải Mô hình ---
#         model = MobileNetV3MultiPalletHeatmap().to(device)

#         if os.path.exists(path_model_center):
#             checkpoint = torch.load(path_model_center, map_location=device)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             print(f"Loaded best model from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.6f})")
#         else:
#             print(
#                 f"Warning: Checkpoint not found at {path_model_center}. Using uninitialized model.")

#         model.eval()
#         pipeline, align = get_image.pre()
#         while True:
#             rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
#             start = time.time()
#             detections = predict_multi_pallet(model, rgb, device)
#             end = time.time()
#             list_time.append(end - start)
#             for detection in detections:
#                 if detection['confidence'] > 0.5:
#                     cv2.circle(rgb, (detection['x'], detection['y']), 4, (0, 255, 0), -1)
#             cv2.imshow('res', rgb)
#             key = cv2.waitKey(1)
#             # Press esc or 'q' to close the image window
#             if key & 0xFF == ord('q') or key == 27:
#                 print("Average inference time:", sum(list_time)/len(list_time))
#                 cv2.destroyAllWindows()
#                 break


def load_model_onnx(path_model_onnx):
    session = ort.InferenceSession(
        path_model_onnx,
        providers=['CPUExecutionProvider']
    )
    return session
if __name__ == "__main__":
    # export_to_onnx(path_model_center, output_path='./new_project/resnet18_multi_center.onnx')
    list_time = []
    pipeline, align = get_image.pre()
    session = load_model_onnx(model_onnx_path)
    while True:
        rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
        start = time.time()
        detections = predict_multi_pallet_onnx(session, rgb)
        end = time.time()
        list_time.append(end - start)
        print("confidence:", detections[0]['confidence'] if len(
            detections) > 0 else "No detections")
        for detection in detections:
            if detection['confidence'] > 0.3:
                cv2.circle(
                    rgb, (detection['x'], detection['y']), 4, (0, 255, 0), -1)
        cv2.imshow('res', rgb)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print("Average inference time:", sum(list_time[1:])/len(list_time))
            cv2.destroyAllWindows()
            break


# folder = r"C:\Lap trinh\realsense\image"
# if __name__ == "__main__":
#     session = load_model_onnx(model_onnx_path)
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
#                     detection['y']), 4, (0, 255, 0), -1)
#             cv2.putText(image, f"{detection['confidence']:.2f}", (detection['x'], detection['y']-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         cv2.imshow('res', image)
#         cv2.waitKey(0)
