# import os
# import random
# import cv2
# import matplotlib.pyplot as plt

# # ====== CẤU HÌNH ======
# # đổi thành đường dẫn của bạn
# image_folder = r"C:\Lap trinh\realsense\output\New folder (2)"
# img_extensions = ('.jpg', '.png', '.jpeg', '.bmp')

# # ====== LẤY DANH SÁCH ẢNH ======
# image_files = [
#     os.path.join(image_folder, f)
#     for f in os.listdir(image_folder)
#     if f.lower().endswith(img_extensions)
# ]

# if len(image_files) < 6:
#     raise ValueError("Folder phải có ít nhất 6 ảnh!")

# # ====== CHỌN NGẪU NHIÊN 6 ẢNH ======
# selected_images = random.sample(image_files, 6)

# # ====== HIỂN THỊ 2x3 ======
# fig, axes = plt.subplots(2, 3, figsize=(12, 8))
# axes = axes.flatten()

# for ax, img_path in zip(axes, selected_images):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     ax.imshow(img)
#     ax.set_title(os.path.basename(img_path), fontsize=9)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

# from ultralytics import YOLO
# import cv2
# import os

# path_model = r'C:\Lap trinh\realsense\project\models\best_ad.pt'

# model = YOLO(path_model, task='detect')

# folder = r"C:\Lap trinh\realsense\image"
# path_save = r"C:\Lap trinh\realsense\output"
# files = []
# for f in os.listdir(folder):
#     if f.lower().endswith(".png"):
#         files.append(os.path.join(folder, f))

# for file in files:
#     image = cv2.imread(file)
#     results = model(image)
#     annotated_frame = results[0].plot()
#     # cv2.imshow("YOLO Detection", annotated_frame)
#     name = os.path.join(path_save, os.path.basename(file))
#     cv2.imwrite(name, annotated_frame)
#     cv2.waitKey(0)


# import cv2
# import os
# from ultralytics import YOLO
# import tracking
# import numpy as np
# import mobilev3ad
# import mobilenetv3_model
# import model_mobilenetv3_hybrid
# import time
# def image_sequence_loader(img_dir):
#     images = sorted(os.listdir(img_dir))
#     for img_name in images[15:]:
#         yield cv2.imread(os.path.join(img_dir, img_name))


# dir = r'C:\Lap trinh\realsense\project\video4\rgb'
# path_model_detection = r"C:\Lap trinh\realsense\final_project\model\best.pt"
# model = YOLO(path_model_detection)
# tracker = tracking.PalletTracker()
# # Ví dụ sử dụng
# model_onnx_path = r'C:\Lap trinh\realsense\final_project\model\onnx_hybrid\model_fast.onnx'
# session = model_mobilenetv3_hybrid.load_model_onnx(model_onnx_path)
# path_save = r"C:\Lap trinh\realsense\output\video_test"

# list_time_tracking = []
# list_time_detect = []

# frame_id = 0

# for frame in image_sequence_loader(dir):
#     print('count', frame_id)
#     image = frame.copy()
#     start_tracking = time.time()
#     tracked_objects = tracking.tracking_phase_ad(model, frame, tracker)  
#     end_tracking = time.time()
#     list_time_tracking.append(end_tracking - start_tracking)
#     image = tracking.visual_tracking(image, tracked_objects)
    
#     # results = model(frame, verbose = False, conf=0.5)
#     # image = results[0].plot()
    
#     # start_detect = time.time()
#     # detections = model_mobilenetv3_hybrid.predict_multi_pallet_onnx(
#     #     session, frame)
#     # print("confidence:", detections[0]['confidence'] if len(
#     #     detections) > 0 else "No detections")
#     # end_detect = time.time()
#     # # print("Time detection: ", end_detect - start_detect)
#     # list_time_detect.append(end_detect - start_detect)
#     # for detection in detections:
#     #     if detection['confidence'] > 0.5:
#     #         cv2.circle(image, (detection['x'], detection['y']), 4, (0, 255, 0), -1)
#     cv2.imshow('Tracking', image)
#     save_name = os.path.join(path_save, f"{frame_id:06d}.png")
    
#     cv2.imwrite(save_name, image)
#     frame_id += 1
#     if cv2.waitKey(1) == 27:
#         break
# cv2.destroyAllWindows()
# print("Average time tracking: ", np.mean(list_time_tracking[1:]))
# print("Average time detection: ", np.mean(list_time_detect[1:]))
# import cv2
# import numpy as np


# def generate_elliptical_heatmap(
#     img_h, img_w,
#     center_x, center_y,
#     sigma_x, sigma_y,
#     stride=1
# ):
#     """
#     Generate elliptical Gaussian heatmap

#     Args:
#         img_h, img_w : image size
#         center_x, center_y : center in image coordinates
#         sigma_x, sigma_y : ellipse std in image space
#         stride : downsample factor (e.g. 4)

#     Returns:
#         heatmap: (H/stride, W/stride)
#     """
#     H = img_h // stride
#     W = img_w // stride

#     cx = center_x / stride
#     cy = center_y / stride

#     xs = np.arange(W)
#     ys = np.arange(H)
#     xv, yv = np.meshgrid(xs, ys)

#     heatmap = np.exp(
#         -(
#             ((xv - cx) ** 2) / (2 * sigma_x ** 2) +
#             ((yv - cy) ** 2) / (2 * sigma_y ** 2)
#         )
#     )

#     return heatmap.astype(np.float32)


# def overlay_heatmap_on_image(
#     image,
#     heatmap,
#     stride=1,
#     alpha=0.5,
#     colormap=cv2.COLORMAP_JET
# ):
#     """
#     Overlay heatmap onto original image

#     Args:
#         image: BGR image (H, W, 3)
#         heatmap: (H/stride, W/stride)
#     """
#     # Resize heatmap to image size
#     heatmap_resized = cv2.resize(
#         heatmap,
#         (image.shape[1], image.shape[0])
#     )

#     heatmap_norm = np.clip(heatmap_resized * 255, 0, 255).astype(np.uint8)
#     heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)

#     overlay = cv2.addWeighted(
#         image, 1 - alpha,
#         heatmap_color, alpha,
#         0
#     )

#     return overlay

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     # Load image
#     img = cv2.imread(r"C:\Lap trinh\realsense\project\images\rgb\0027.png")
#     img_h, img_w = img.shape[:2]

#     # Center pallet
#     center_x = 280
#     center_y = 280

#     # Ellipse shape (ví dụ pallet rộng ngang)
#     sigma_x = 5   # rộng
#     sigma_y = 3   # thấp

#     # Generate heatmap
#     heatmap = generate_elliptical_heatmap(
#         img_h, img_w,
#         center_x, center_y,
#         sigma_x, sigma_y,
#         stride=4
#     )

#     # Overlay
#     overlay = overlay_heatmap_on_image(
#         img,
#         heatmap,
#         stride=4,
#         alpha=0.3
#     )

#     # Visualize
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
#     plt.title("Overlay heatmap")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()

# import os

# src_dir = r"C:\Lap trinh\blender_proc\dataset\val\labels"  # thư mục label gốc
# dst_dir = r"C:\Lap trinh\blender_proc\dataset\val\labels_yolo"  # thư mục lưu label YOLO

# # Tạo thư mục đích nếu chưa tồn tại
# os.makedirs(dst_dir, exist_ok=True)

# for filename in os.listdir(src_dir):
#     if not filename.endswith(".txt"):
#         continue

#     src_path = os.path.join(src_dir, filename)
#     dst_path = os.path.join(dst_dir, filename)

#     with open(src_path, "r") as f:
#         line = f.readline().strip()

#     # Bỏ file rỗng
#     if not line:
#         print(f"[WARNING] Empty file: {filename}")
#         continue

#     parts = line.split()

#     # Kiểm tra đủ dữ liệu
#     if len(parts) < 5:
#         print(f"[ERROR] Invalid label format: {filename}")
#         continue

#     # Lấy đúng 5 phần đầu cho YOLO
#     yolo_line = " ".join(parts[:5])

#     # Ghi ra file mới
#     with open(dst_path, "w") as f:
#         f.write(yolo_line + "\n")

#     print(f"[OK] {filename}")


import cv2
import os

# ===== CẤU HÌNH =====
image_folder = r"C:\Lap trinh\realsense\output\video_test"        # thư mục chứa ảnh
output_video = "output.mp4"     # tên video xuất ra
fps = 15

# ===== LẤY DANH SÁCH ẢNH =====
images = sorted([
    img for img in os.listdir(image_folder)
    if img.endswith(('.png', '.jpg', '.jpeg'))
])

assert len(images) > 0, "Folder không có ảnh!"

# ===== ĐỌC ẢNH ĐẦU TIÊN LẤY KÍCH THƯỚC =====
first_img_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_img_path)
h, w, _ = frame.shape

# ===== KHỞI TẠO VIDEO WRITER =====
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # hoặc 'XVID'
video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# ===== GHI TỪNG FRAME =====
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Bỏ qua ảnh lỗi: {img_name}")
        continue

    # đảm bảo cùng kích thước
    frame = cv2.resize(frame, (w, h))
    video.write(frame)

video.release()
print("✅ Hoàn thành tạo video:", output_video)
