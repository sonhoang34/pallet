# import numpy as np
# import cv2
# import time
# import model_yolo
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import get_image

# # ===== CONFIG =====
# path_image = r"C:\Lap trinh\realsense\project\images\rgb\0006.png"
# path_model_yolo = r'C:\Lap trinh\realsense\final_project\model\best.pt'

# model = YOLO(path_model_yolo)

# width = 640
# heigh = 480
# fx, fy, cx, cy = [609.4961547851562, 609.9985961914062,
#                   324.0486145019531, 231.71307373046875]

# K = np.array([
#     [fx, 0.0, cx],
#     [0.0, fy, cy],
#     [0.0, 0.0, 1.0]
# ])

# y_ground = 0.3   # mét (camera cao 30 cm)

# # function=============================

# def pixel_to_ray(u, v, K):
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]

#     x = (u - cx) / fx
#     y = (v - cy) / fy

#     d = np.array([x, y, 1.0])
#     return d / np.linalg.norm(d)


# def intersect_ground(ray_dir, y_ground):
#     dy = ray_dir[1]
#     if abs(dy) < 1e-6:
#         return None

#     t = y_ground / dy
#     if t <= 0:
#         return None

#     x = t * ray_dir[0]
#     z = t * ray_dir[2]
#     return np.array([x, z])


# def fit_line_ransac(points, threshold=0.005, iters=50):
#     best_inliers = []
#     best_model = None

#     N = len(points)
#     if N < 2:
#         return None, None
#     for _ in range(iters):
#         i, j = np.random.choice(N, 2, replace=False)
#         p1, p2 = points[i], points[j]

#         d = p2 - p1
#         d /= np.linalg.norm(d)

#         # line normal
#         n = np.array([-d[1], d[0]])

#         dist = np.abs((points - p1) @ n)
#         inliers = np.where(dist < threshold)[0]

#         if len(inliers) > len(best_inliers):
#             best_inliers = inliers
#             best_model = (p1, d)

#     return best_model, points[best_inliers]

# def find_long_edge(rgb, bbox):

#     # YOLO detect (ngoài timing nếu cần)
#     x_min, y_min, x_max, y_max = bbox
#     y_min = y_min + 10
#     y_max = y_max + 10

#     # start = time.time()

#     # 1️⃣ ROI
#     roi = rgb[y_min:y_max, x_min:x_max]

#     # 2️⃣ Gray + Blur
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 3️⃣ Sobel Y (16-bit là đủ)
#     sobely = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
#     # edge = cv2.convertScaleAbs(sobely)

#     _, edge = cv2.threshold(sobely, 50, 255, cv2.THRESH_BINARY)
#     edge = edge.astype(np.uint8)

#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

#     # plt.imshow(edge)
#     # plt.show()
# # # 4️⃣ Contours (giảm số điểm)
#     contours, _ = cv2.findContours(
#         edge,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     if not contours:
#         raise RuntimeError("No contour found")

#     longest = max(contours, key=lambda c: cv2.arcLength(c, False))

#     pts = longest.reshape(-1, 2)

#     # PCA
#     mean = pts.mean(axis=0)
#     _, _, Vt = np.linalg.svd(pts - mean)
#     direction = Vt[0]              # (2,)
#     direction /= np.linalg.norm(direction)

#     # project scalar
#     t = (pts - mean) @ direction

#     t_min, t_max = np.percentile(t, [5, 95])   # 🔥 bỏ outlier

#     # điểm chính xác trên trục PCA
#     p_start = mean + t_min * direction
#     p_end = mean + t_max * direction

#     p_start = tuple(p_start.astype(int))
#     p_end = tuple(p_end.astype(int))

#     print(p_start)
#     print(p_end)
#     # 6️⃣ ROI → ảnh gốc
#     p_start_origin = (int(p_start[0] + x_min), int(p_start[1] + y_min))
#     p_end_origin = (int(p_end[0] + x_min), int(p_end[1] + y_min))

#     # 7️⃣ Lấy 50 điểm đều nhau
#     num_points = 100
#     xs = np.linspace(p_start_origin[0],
#                      p_end_origin[0], num_points).astype(int)
#     ys = np.linspace(p_start_origin[1],
#                      p_end_origin[1], num_points).astype(int)
#     return (xs, ys)

# def yaw_predict(xs, ys, K, y_ground):
#     points_xz = []
#     for x, y in zip(xs, ys):
#         # cv2.circle(image_vis, (x, y), 2, (0, 0, 255), -1)
#         ray = pixel_to_ray(x, y, K)
#         p = intersect_ground(ray, y_ground)
#         if p is not None:
#             points_xz.append(p)
#     points_xz = np.asarray(points_xz)

#     (line_p, line_d), inliers = fit_line_ransac(points_xz)

#     # ép hướng vector
#     if line_d[0] < 0:      # line_d = [dx, dz]
#         line_d = -line_d

#     # print(line_d)
#     yaw = np.arctan2(line_d[0], line_d[1])  - 1.5708  # yaw trên mặt sàn
#     # print(f'Theta: {(np.rad2deg(yaw) - 90):.2f}')
    
#     return yaw
# # ==================++++++++++++++++++++

# if __name__ == "__main__":
#     image = cv2.imread(path_image)
#     x_min, y_min, x_max, y_max, conf = model_yolo.detect_pallet(
#         model, image)
#     image_vis = image.copy()
#     bbox = [x_min, y_min, x_max, y_max]
#     start = time.time()
#     xs, ys = find_long_edge(image, bbox)
#     yaw = yaw_predict(xs, ys, K, y_ground)
#     end = time.time()
#     print(end - start)
    
#     print('yaw: ', round(np.rad2deg(yaw), 2))
#     for x, y in zip(xs, ys):
#         cv2.circle(image_vis, (x, y), 2, (0, 0, 255), -1)
#     plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.show()

# # if __name__ == '__main__':
# #     pipeline, align = get_image.pre()
# #     list_time = []
# #     while True:
# #         rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
# #         x_min, y_min, x_max, y_max, conf = model_yolo.detect_pallet(
# #             model, rgb)

# #         bbox = [x_min, y_min, x_max, y_max]
        
# #         image_vis = rgb.copy()
# #         if bbox is not None and not all(v == 0 for v in bbox):
# #             start = time.time()
# #             xs, ys = find_long_edge(rgb, bbox)
# #             yaw = yaw_predict(xs, ys, K, y_ground)
# #             end = time.time()
# #             list_time.append(end - start)
# #             print(end - start)
# #             print('yaw: ', yaw)
# #             for x, y in zip(xs, ys):
# #                 cv2.circle(image_vis, (x, y), 2, (0, 0, 255), -1)
        
               
# #         # cv2.rectangle(image_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
# #         cv2.imshow('res', image_vis)
        
# #         key = cv2.waitKey(1)
# #         # Press esc or 'q' to close the image window
# #         if key & 0xFF == ord('q') or key == 27:
# #             print(np.mean(list_time[1:]))
# #             cv2.destroyAllWindows()
# #             break
        

import numpy as np
import cv2
import time
import model_yolo
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ================= CONFIG =================
path_image = r"C:\Lap trinh\realsense\project\images\rgb\0006.png"
path_model_yolo = r'C:\Lap trinh\realsense\final_project\model\best.pt'

model = YOLO(path_model_yolo)

fx, fy, cx, cy = [609.4961547851562, 609.9985961914062,
                  324.0486145019531, 231.71307373046875]

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])

y_ground = 0.3  # camera cao 30cm

# ================= GEOMETRY =================


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


# ================= RANSAC LINE =================
def fit_line_ransac(points, threshold=0.01, iters=50):
    best_score = -1
    best_model = None
    best_inliers = None

    N = len(points)
    if N < 2:
        return None, None

    for _ in range(iters):
        i, j = np.random.choice(N, 2, replace=False)
        p1, p2 = points[i], points[j]

        d = p2 - p1
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            continue
        d /= norm

        n = np.array([-d[1], d[0]])
        dist = np.abs((points - p1) @ n)
        inliers = np.where(dist < threshold)[0]

        score = len(inliers)
        if score > best_score:
            best_score = score
            best_model = (p1, d)
            best_inliers = inliers

    return best_model, points[best_inliers]


# ================= EDGE EXTRACTION =================
def find_long_edge(rgb, bbox):
    x_min, y_min, x_max, y_max = bbox
    roi = rgb[y_min:y_max, x_min:x_max]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sobely = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
    _, edge = cv2.threshold(sobely, 50, 255, cv2.THRESH_BINARY)
    edge = edge.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    longest = max(contours, key=lambda c: cv2.arcLength(c, False))
    pts = longest.reshape(-1, 2)

    # PCA để lấy trục chính
    mean = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mean)
    direction = Vt[0]
    direction /= np.linalg.norm(direction)

    t = (pts - mean) @ direction
    t_min, t_max = np.percentile(t, [5, 95])

    p_start = mean + t_min * direction
    p_end = mean + t_max * direction

    p_start = p_start.astype(int)
    p_end = p_end.astype(int)

    # về ảnh gốc
    p_start = (p_start[0] + x_min, p_start[1] + y_min)
    p_end = (p_end[0] + x_min,   p_end[1] + y_min)

    xs = np.linspace(p_start[0], p_end[0], 100).astype(int)
    ys = np.linspace(p_start[1], p_end[1], 100).astype(int)

    return xs, ys


# ================= YAW ESTIMATION (ΔZ PENALTY) =================
def yaw_predict(xs, ys, K, y_ground):
    points_xz = []

    for x, y in zip(xs, ys):
        ray = pixel_to_ray(x, y, K)
        p = intersect_ground(ray, y_ground)
        if p is not None:
            points_xz.append(p)

    points_xz = np.asarray(points_xz)
    if len(points_xz) < 2:
        return None, None

    # ---- ΔZ penalty ----
    z_start = points_xz[0, 1]
    z_end = points_xz[-1, 1]
    delta_z = abs(z_end - z_start)

    L = np.linalg.norm(points_xz[-1] - points_xz[0])
    tilt_ratio = delta_z / (L + 1e-6)   # ≈ tan(pitch)

    # ---- fit line ----
    model, inliers = fit_line_ransac(points_xz)
    if model is None:
        return None, tilt_ratio

    _, d = model
    if d[0] < 0:
        d = -d

    yaw = np.arctan2(d[0], d[1]) - np.pi / 2
    return yaw, tilt_ratio


# ================= MAIN =================
if __name__ == "__main__":
    image = cv2.imread(path_image)
    image_vis = image.copy()

    x_min, y_min, x_max, y_max, conf = model_yolo.detect_pallet(model, image)
    bbox = [x_min, y_min, x_max, y_max]

    xs, ys = find_long_edge(image, bbox)
    if xs is None:
        print("No edge found")
        exit()

    start = time.time()
    yaw, tilt_ratio = yaw_predict(xs, ys, K, y_ground)
    end = time.time()

    print(f"time: {end - start:.4f}s")
    print(f"yaw (deg): {np.rad2deg(yaw):.2f}")
    print(f"tilt_ratio (≈tan pitch): {tilt_ratio:.3f}")

    for x, y in zip(xs, ys):
        cv2.circle(image_vis, (x, y), 2, (0, 0, 255), -1)

    plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
