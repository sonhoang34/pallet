import get_image
import cv2
from ultralytics import YOLO
import open3d as o3d
import numpy as np
import math
import tracking
import time
import func
import model_mobilenetv3_hybrid
# import edge_detection
import rot_detection

path_model_yolo = r'C:\Lap trinh\realsense\final_project\model\yolo_model\best_2.pt'
path_model_mobilenet_center = r'C:\Lap trinh\realsense\final_project\model\onnx_hybrid\model_fast.onnx'
model_onnx_path = r"C:\Lap trinh\realsense\final_project\model\onnx_hybrid_2\model_fast.onnx"
width = 640
heigh = 480
fx, fy, cx, cy = [609.4961547851562, 609.9985961914062,
                  324.0486145019531, 231.71307373046875]
K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])

y_ground = 0.27   # mét (camera cao 30 cm)

def load_models(path_model_yolo, path_model_mobilenet_center):
    model_yolo = YOLO(path_model_yolo, task='detect')
    print("Load YOLO model done.")
    model_center = model_mobilenetv3_hybrid.load_model_onnx(path_model_mobilenet_center)
    print("Load MobileNetV3 model done.")
    return model_yolo, model_center

def solution_ad(rgb, depth_raw, model_yolo, model_center, tracker):
    img = rgb.copy()
    
    # Step 1: Detect pallet using YOLO & visualize
    start_detection = time.time()
    tracking_objects = tracking.tracking_phase_ad(model_yolo, rgb, tracker)
    end_detection = time.time()
    # print("Time detection: ", end_detection - start_detection)
    tracking.visual_tracking(img, tracking_objects)

    x, y = None, None
    bbox = list(map(round, tracking_objects[0][0])) if len(tracking_objects) > 0 else None
    
    rgb, roi = rot_detection.gamma_roi(rgb, bbox)
    
    if bbox is not None:
        start_center = time.time()
        detections = model_mobilenetv3_hybrid.predict_multi_pallet_onnx(
            model_center, rgb, detection_threshold=0.3)
        end_center = time.time()
        # print("Time center detection: ", end_center - start_center)
        for detection in detections:

            if detection['confidence'] > 0.3 and func.is_in_bbox(bbox, detection['x'], detection['y']) and depth_raw[detection['y'], detection['x']] > 0:
                # print(func.is_in_bbox(bbox, detection['x']), depth_raw[detection['y'], detection['x']])
                x = detection['x']
                y = detection['y']  # offset a bit to center

        if x is None or y is None:
            return img
        
        # func.visual(img, fx, fy, cx, cy, points_2d=(x, y))
        
        dis = func.find_distance_ad(x, y, fx, fy, cx, cy, depth_raw)
        
        xs, ys = rot_detection.detect_pallet_edge_hough(roi, bbox)
        rot = rot_detection.rot_predict(xs, ys, K, y_ground)
        
        func.visual(img, fx, fy, cx, cy, theta=rot, pos_3d=dis)
    return img   


# def solution(rgb, depth_raw, model_yolo, model_center, tracker):
#     img = rgb.copy()
    
#     # Step 1: Detect pallet using YOLO & visualize
#     start_detection = time.time()
#     tracking_objects = tracking.tracking_phase(model_yolo, rgb, tracker)
#     end_detection = time.time()
#     print("Time detection: ", end_detection - start_detection)
#     tracking.visual_tracking(img, tracking_objects)
    
#     x, y = None, None
#     bbox = list(map(round, tracking_objects[0][0])) if len(tracking_objects) > 0 else None
#     # print("bbox:", bbox)
#     if bbox is not None:
#         start_center = time.time()
#         detections = mobilev3ad.predict_multi_pallet_onnx(model_center, rgb, detection_threshold=0.3)
#         end_center = time.time()
#         print("Time center detection: ", end_center - start_center)
#         # print((detections))
#         for detection in detections:
            
#             if detection['confidence'] > 0.3 and func.is_in_bbox(bbox, detection['x'], detection['y']) and depth_raw[detection['y'], detection['x']] > 0:
#                 # print(func.is_in_bbox(bbox, detection['x']), depth_raw[detection['y'], detection['x']])
#                 x = detection['x']
#                 y = detection['y']  # offset a bit to center
        
#         if x is None or y is None:
#             return img
        
#         # print(x, y)
#         func.visual(img, fx, fy, cx, cy, points_2d=(x, y))
        
#         center_point_pixel_depth = [[x, y, depth_raw[y, x]]]
#         point_center_3d = func.convert_2d_to_3d_coordinates(center_point_pixel_depth, fx, fy, cx, cy)
#         points_bbox_pixel_depth = func.find_all_2d_points_in_bounding_box(bbox, depth_raw, stride=2)
#         points_3d = func.convert_2d_to_3d_coordinates(points_bbox_pixel_depth, fx, fy, cx, cy)
        
#         if len(points_3d) > 250:
#             # Tao du lieu point cloud
#             pcd_pallet = o3d.geometry.PointCloud()
#             pcd_pallet.points = o3d.utility.Vector3dVector(points_3d)
#             filter_u = depth_raw[y, x] - 200
#             filter_a = depth_raw[y, x] + 200
#             mask = (points_3d[:, 2] >= filter_u) & (points_3d[:, 2] <= filter_a)
#             pcd_pallet = pcd_pallet.select_by_index(np.where(mask)[0])
            
#             # Tim tap diem point cloud tao thanh mat truoc pallet
#             if len(pcd_pallet.points) > 100:
#                 front_plane_points, plane_model = func.extract_pallett_front_points(pcd_pallet, 15, 500)
#                 if len(front_plane_points.points) > 50:
#                     n = plane_model[0:3]
#                     n_unit = n / np.linalg.norm(n)
#                     if (abs(n_unit[1]) < 0.5):
#                         normal_v = [n[0], 0, n[2]]
#                         normal_v_unit = normal_v / np.linalg.norm(normal_v)
#                         theta_rad = math.atan2(normal_v_unit[0], normal_v_unit[2])
#                         func.visual(img, fx, fy, cx, cy,
#                                     theta=theta_rad, pos_3d=point_center_3d)
#     return img

if __name__ == "__main__":
    tracker = tracking.PalletTracker()
    
    model_yolo, model_center = load_models(
        path_model_yolo, path_model_mobilenet_center)
    
    pipeline, align = get_image.pre()
    
    list_time = []
    while True:
        rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
        start = time.time()
        result = solution_ad(rgb, depth_raw, model_yolo, model_center, tracker)
        end = time.time()
        list_time.append(end - start)
        # print(end - start)
        
        cv2.imshow('result', result)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print('Average time:', np.mean(list_time[1:]))
            print("FPS: ", round(1/np.mean(list_time[1:])))
            cv2.destroyAllWindows()
            break
    

