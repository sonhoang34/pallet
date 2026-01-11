import cv2
import numpy as np
import time
import model_yolo
import model_mobilenetv3_hybrid
from ultralytics import YOLO

path_image = r"C:\Lap trinh\realsense\project\images\rgb\0047.png"
path_model_yolo = r'C:\Lap trinh\realsense\final_project\model\yolo_model\best_1.pt'
path_model_center = r'C:\Lap trinh\realsense\final_project\model\onnx_hybrid_2\model_fast.onnx'

width = 640
heigh = 480
fx, fy, cx, cy = [609.4961547851562, 609.9985961914062,
                  324.0486145019531, 231.71307373046875]

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])

y_ground = 0.30   # mét (camera cao 30 cm)

def gamma_correction(image, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


def gamma_roi(image, bbox, gamma=0.5):
    x, y, x_, y_ = bbox
    out = image.copy()
    roi_img = out[y:y_, x:x_]
    out[y:y_, x:x_] = gamma_correction(roi_img, gamma)
    return out, roi_img

def detect_pallet_edge_hough(roi, bbox):

    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(roi, 5, 75, 75)
    # blur = cv2.GaussianBlur(roi, (3, 3), 0)
    # 1. Edge detection
    edges = cv2.Canny(blur, 50, 150)
    x_min, y_min = bbox[0], bbox[1] + 10
    # 2. Morphology để kết nối
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(edges)
    # plt.show()
    # 3. Hough Lines (probabilistic - faster)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=20
    )

    # print(lines)
    if lines is None:
        return None

    # 4. Filter horizontal lines (edge đáy)
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))

        if angle < np.deg2rad(15):  # < 15° = horizontal
            horizontal_lines.append(line[0])

    if not horizontal_lines:
        return None

    # 5. Chọn line dài nhất
    longest = max(horizontal_lines,
                  key=lambda l: np.hypot(l[2]-l[0], l[3]-l[1]))

    p_start_origin = (longest[0] + x_min, longest[1] + y_min)
    p_end_origin = (longest[2] + x_min, longest[3] + y_min)
    
    num_points = 20
    x_vals = np.linspace(p_start_origin[0], p_end_origin[0], num_points)
    y_vals = np.linspace(p_start_origin[1], p_end_origin[1], num_points)
    return (x_vals, y_vals)

def pixel_to_ray(u, v, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy
    d = np.array([x, y, 1.0])
    return d / np.linalg.norm(d)


def intersect_ground(ray_dir, y_ground):
    dy = ray_dir[1]
    if abs(dy) < 1e-6:
        return None

    t = y_ground / dy
    if t <= 0:
        return None

    x = t * ray_dir[0]
    z = t * ray_dir[2]
    return np.array([x, z])

def fit_line_ransac(points, threshold=0.005, iters=50):
    best_inliers = []
    best_model = None

    N = len(points)
    if N < 2:
        return None, None
    for _ in range(iters):
        i, j = np.random.choice(N, 2, replace=False)
        p1, p2 = points[i], points[j]

        d = p2 - p1
        d /= np.linalg.norm(d)

        # line normal
        n = np.array([-d[1], d[0]])

        dist = np.abs((points - p1) @ n)
        inliers = np.where(dist < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (p1, d)

    return best_model, points[best_inliers]

def rot_predict(xs, ys, K, y_ground):
    
    points_xz = []
    for x, y in zip(xs, ys):
        # cv2.circle(image_vis, (x, y), 2, (0, 0, 255), -1)
        ray = pixel_to_ray(x, y, K)
        p = intersect_ground(ray, y_ground)
        if p is not None:
            points_xz.append(p)
    points_xz = np.asarray(points_xz)

    (line_p, line_d), inliers = fit_line_ransac(points_xz)

    # ép hướng vector
    if line_d[0] < 0:      # line_d = [dx, dz]
        line_d = -line_d

    # print(line_d)
    rot = np.arctan2(line_d[0], line_d[1])  - np.pi /2  # yaw trên mặt sàn
    # print(f'Theta: {(np.rad2deg(yaw) - 90):.2f}')

    return rot

if __name__ == "__main__":
    
    image = cv2.imread(path_image)
    model = model_yolo.load_model(path_model_yolo)
    x_min, y_min, x_max, y_max, _ = model_yolo.detect_pallet(model, image)
    bbox = [x_min, y_min + 10, x_max, y_max + 10]
    image_after, roi = gamma_roi(image, bbox, gamma=1.2)
    # cv2.imshow('roi', roi)
    start = time.time()
    
    (xs, ys) = detect_pallet_edge_hough(roi, bbox)
    
    rot = rot_predict(xs, ys, K, y_ground)

    end = time.time()
    print(np.rad2deg(rot))
    print(f"Time taken for gamma correction: {end - start} seconds")

    cv2.imshow("Original Image", image)
    cv2.imshow("Gamma Corrected Image", image_after)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    