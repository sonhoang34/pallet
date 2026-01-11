import numpy as np
import cv2


def median_point(points):
    """
    points: ndarray shape (N, 3)
    return: ndarray shape (3,)
    """
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 3
    return np.median(points, axis=0)

def find_distance(x, y, fx, fy, cx, cy, depth_raw):
    center = [x, y, depth_raw[y, x]]
    point1 = [x, y - 5, depth_raw[y-5, x]]
    point2 = [x + 5, y, depth_raw[y, x+5]]
    point3 = [x, y + 5, depth_raw[y+5, x]]
    point4 = [x - 5, y, depth_raw[x-5, y]]
    
    points = [center, point1, point2, point3, point4]
    
    points_3d = convert_2d_to_3d_coordinates(points, fx, fy, cx, cy)
    
    points_3d = median_point(points_3d)
    
    return [list(points_3d)]
    

def find_distance_ad(x, y, fx, fy, cx, cy, depth_raw):
    # Tạo offsets cho 5 điểm (center + 4 hướng)
    offsets = np.array([[0, 0], [0, -5], [5, 0], [0, 5], [-5, 0]])

    # Tính tọa độ 2D của các điểm
    coords = np.array([x, y]) + offsets

    # Lấy depth values (sửa bug: đúng thứ tự y, x)
    depths = depth_raw[coords[:, 1], coords[:, 0]]

    # Tạo points 2D+depth
    points = np.column_stack([coords, depths])

    # Convert sang 3D và tính median
    points_3d = convert_2d_to_3d_coordinates(points, fx, fy, cx, cy)

    return np.median(points_3d, axis=0)

def find_all_2d_points_in_bounding_box(box, depth_align, stride=5):
    x1, y1, x2, y2 = map(round, box)
    # depth_raw = np.asarray(o3d.io.read_image(depth_path))
    # Cắt depth ROI trực tiếp
    depth_crop = depth_align[y1:y2, x1:x2]

    ys, xs = np.mgrid[y1:y2:stride, x1:x2:stride]
    zs = depth_crop[::stride, ::stride]

    # Ghép lại thành (N, 3)
    points_2d = np.column_stack((xs.ravel(), ys.ravel(), zs.ravel()))
    return points_2d


def convert_2d_to_3d_coordinates(points_2d, fx, fy, cx, cy):
    if points_2d is None or len(points_2d) == 0:
        return []

    points_array = np.array(points_2d)

    # Lọc điểm có depth > 0
    valid_mask = points_array[:, 2] > 0
    valid_points = points_array[valid_mask]

    if valid_points.shape[0] == 0:
        return []

    x_2d = valid_points[:, 0]
    y_2d = valid_points[:, 1]
    z = valid_points[:, 2]

    x_3d = (x_2d - cx) * (z / fx)
    y_3d = (y_2d - cy) * (z / fy)

    points_3d = np.column_stack([x_3d, y_3d, z])
    return points_3d


def extract_pallett_front_points(point_cloud, threshold, num_iter):
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=threshold,  # khoảng cách tối đa từ mặt phẳng dv mm
        ransac_n=3,               # 3 điểm để xác định plane
        num_iterations=num_iter     # số vòng thử RANSAC
    )
    front_plane_points = point_cloud.select_by_index(inliers)
    return front_plane_points, plane_model


def visual(img, fx, fy, cx, cy, bbox=None, theta=None, points_2d=None, pos_3d=None):
    if bbox != None : 
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
    if points_2d != None:
        cv2.circle(img, (points_2d[0], points_2d[1]), 4, (0, 0, 255), -1)
    

    if theta is not None and pos_3d is not None:
        if len(pos_3d) == 0 or np.any(np.isnan(pos_3d)):
            pass
        else:
            rvec = np.array([[0.0], [theta], [0.0]], dtype=np.float64)
            tvec = np.array(pos_3d, dtype=np.float64).reshape(3, 1)

            axis_points_3d = np.float32([
                [0, 0, 0],
                [100, 0, 0],
                [0, 100, 0],
                [0, 0, 100]
            ])

            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            imgpts, _ = cv2.projectPoints(
                axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
            o = tuple(imgpts[0].ravel().astype(int))
            x = tuple(imgpts[1].ravel().astype(int))
            y = tuple(imgpts[2].ravel().astype(int))
            z = tuple(imgpts[3].ravel().astype(int))
            cv2.arrowedLine(img, o, z, (255, 0, 0), 2)
            cv2.arrowedLine(img, o, x, (0, 0, 255), 2)
            cv2.arrowedLine(img, o, y, (0, 255, 0), 2)
            theta_deg = np.rad2deg(theta)
            cv2.putText(img, f"Dis z: {pos_3d[0][2]:.2f} Dis x: {pos_3d[0][0]:.2f}", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Rot: {theta_deg:.2f}", (40, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Dis z: {pos_3d[0][2]:.2f} Dis x: {pos_3d[0][0]:.2f}")
            print(f"Theta: {theta_deg:.2f}")
    
def is_in_bbox(bbox, x_center, y_center):
    x_min, x_max = bbox[0], bbox[2]
    y_min, y_max = bbox[1], bbox[3]
    if ((x_center - x_min) >= 0.3*(x_max - x_min)) and \
        ((x_max - x_center) <= 0.7*(x_max - x_min)
         and (x_center < x_max) and (x_center > x_min)) and \
                (y_center < y_max) and (y_center > y_min):
        return True
    else:
        return False
        

# def is_in_bbox(bbox, x_center):
#     x_min, x_max = bbox[0], bbox[2]
#     if x_center < x_max and x_center > x_min:
#         return True
#     else:
#         return False
