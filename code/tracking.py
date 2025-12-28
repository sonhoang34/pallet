import numpy as np
from collections import deque
import time
import get_image
import cv2
from ultralytics import YOLO
import time


class PalletTracker:
    def __init__(self, iou_threshold=0.3, max_age=5, history_size=50,
                 smart_center_threshold=50, smart_area_threshold=0.3,
                 history_ttl=50, use_hungarian=False):
        """
        Args:
            iou_threshold: Ngưỡng IoU để match
            max_age: Số frame không match thì xóa track
            history_size: Số track lưu trong history
            smart_center_threshold: Khoảng cách center tối đa (pixels)
            smart_area_threshold: % chênh lệch diện tích cho phép
            history_ttl: Số frame history còn hợp lệ cho SmartID
            use_hungarian: True = Hungarian, False = Greedy (nhanh hơn)
        """
        self.tracks = {}  # {id: {'bbox', 'center', 'area', 'age', 'hits'}}
        self.next_id = 1
        self.empty_frames = 0
        self.history = deque(maxlen=history_size)

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.smart_center_threshold = smart_center_threshold
        self.smart_area_threshold = smart_area_threshold
        self.history_ttl = history_ttl
        self.use_hungarian = use_hungarian

    def reset(self):
        """Reset toàn bộ tracking state"""
        self.tracks.clear()
        self.next_id = 1
        self.empty_frames = 0
        self.history.clear()

    def calculate_iou(self, box1, box2):
        """Tính IoU giữa 2 boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def get_center_and_area(self, bbox):
        """Tính center và diện tích của bbox"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return (cx, cy), area

    def calculate_center_distance(self, center1, center2):
        """Tính khoảng cách Euclidean giữa 2 centers"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def smart_id_match(self, center, area):
        """
        Tìm ID phù hợp từ history dựa vào center và area
        Chỉ xét history còn mới (trong history_ttl frames)
        Returns: track_id hoặc None
        """
        best_match = None
        best_score = float('inf')

        for hist_track in self.history:
            # Chỉ xét history còn mới
            if hist_track['frames_ago'] > self.history_ttl:
                continue

            hist_center = hist_track['center']
            hist_area = hist_track['area']

            # Tính khoảng cách center
            center_dist = self.calculate_center_distance(center, hist_center)

            # Tính % chênh lệch diện tích
            area_diff = abs(area - hist_area) / max(area, hist_area)

            # Kiểm tra threshold
            if (center_dist < self.smart_center_threshold and
                    area_diff < self.smart_area_threshold):
                # Score tổng hợp (càng nhỏ càng tốt)
                score = center_dist + area_diff * 100
                if score < best_score:
                    best_score = score
                    best_match = hist_track['id']

        return best_match

    def greedy_match(self, detections, track_ids):
        """
        Greedy matching dựa trên IoU + center distance
        Nhanh hơn Hungarian cho số lượng object nhỏ
        """
        det_centers = [self.get_center_and_area(det)[0] for det in detections]

        matches = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(track_ids)))

        # Tính cost matrix: IoU + center distance
        costs = []
        for i, det in enumerate(detections):
            det_center = det_centers[i]
            for j, tid in enumerate(track_ids):
                track = self.tracks[tid]
                iou = self.calculate_iou(det, track['bbox'])
                center_dist = self.calculate_center_distance(
                    det_center, track['center'])

                # Cost = (1 - IoU) + normalized distance
                cost = (1 - iou) + 0.001 * center_dist
                costs.append((cost, i, j))

        # Sắp xếp theo cost tăng dần
        costs.sort()

        # Greedy matching
        for cost, det_idx, track_idx in costs:
            if det_idx in unmatched_dets and track_idx in unmatched_tracks:
                # Chỉ match nếu IoU đủ tốt
                iou = self.calculate_iou(detections[det_idx],
                                         self.tracks[track_ids[track_idx]]['bbox'])
                if iou >= self.iou_threshold:
                    matches.append((det_idx, track_idx))
                    unmatched_dets.remove(det_idx)
                    unmatched_tracks.remove(track_idx)

        return matches, list(unmatched_dets), list(unmatched_tracks)

    def hungarian_match(self, detections, track_ids):
        """Hungarian matching với cost = IoU + center distance"""
        from scipy.optimize import linear_sum_assignment

        det_centers = [self.get_center_and_area(det)[0] for det in detections]
        cost_matrix = np.zeros((len(detections), len(track_ids)))

        for i, det in enumerate(detections):
            det_center = det_centers[i]
            for j, tid in enumerate(track_ids):
                track = self.tracks[tid]
                iou = self.calculate_iou(det, track['bbox'])
                center_dist = self.calculate_center_distance(
                    det_center, track['center'])

                # Cost = (1 - IoU) + normalized distance
                cost_matrix[i, j] = (1 - iou) + 0.001 * center_dist

        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(track_ids)))

        for det_idx, track_idx in zip(det_indices, track_indices):
            # Chỉ match nếu IoU đủ tốt
            iou = self.calculate_iou(detections[det_idx],
                                     self.tracks[track_ids[track_idx]]['bbox'])
            if iou >= self.iou_threshold:
                matches.append((det_idx, track_idx))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(track_idx)

        return matches, list(unmatched_dets), list(unmatched_tracks)

    def update(self, detections):
        """
        Update tracker với detections mới
        Args:
            detections: List of bboxes [[x1,y1,x2,y2], ...]
        Returns:
            List of (bbox, track_id)
        """
        # Không có detection
        if len(detections) == 0:
            self.empty_frames += 1

            # Tăng age cho tất cả tracks / moi bbox la 1 track
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['age'] += 1

                # Xóa track quá cũ và lưu vào history
                if self.tracks[tid]['age'] > self.max_age:
                    self.history.append({
                        'id': tid,
                        'center': self.tracks[tid]['center'],
                        'area': self.tracks[tid]['area'],
                        'frames_ago': 0
                    })
                    del self.tracks[tid]

            # Reset nếu quá nhiều frame rỗng
            if self.empty_frames >= 45:
                print(f"Reset tracker sau {self.empty_frames} frames rỗng")
                self.reset()

            # Tăng frames_ago cho history
            for hist in self.history:
                hist['frames_ago'] += 1

            return []

        # Reset empty counter
        self.empty_frames = 0

        # Tăng frames_ago cho history
        for hist in self.history:
            hist['frames_ago'] += 1

        # Nếu chưa có track nào, tạo mới
        if len(self.tracks) == 0:
            results = []
            for det in detections:
                center, area = self.get_center_and_area(det)

                # Thử SmartID trước
                matched_id = self.smart_id_match(center, area)

                if matched_id is not None:
                    track_id = matched_id
                    print(f"SmartID: Khôi phục ID {track_id}")
                else:
                    track_id = self.next_id
                    self.next_id += 1

                self.tracks[track_id] = {
                    'bbox': det,
                    'center': center,
                    'area': area,
                    'age': 0,
                    'hits': 1
                }
                results.append((det, track_id))

            return results

        # Matching
        track_ids = list(self.tracks.keys())

        # Chọn thuật toán matching
        if self.use_hungarian or len(detections) * len(track_ids) <= 100:
            matches, unmatched_dets, unmatched_tracks = self.hungarian_match(
                detections, track_ids)
        else:
            matches, unmatched_dets, unmatched_tracks = self.greedy_match(
                detections, track_ids)

        # Update matched tracks
        for det_idx, track_idx in matches:
            track_id = track_ids[track_idx]
            det = detections[det_idx]
            center, area = self.get_center_and_area(det)

            self.tracks[track_id]['bbox'] = det
            self.tracks[track_id]['center'] = center
            self.tracks[track_id]['area'] = area
            self.tracks[track_id]['age'] = 0  # Reset age
            self.tracks[track_id]['hits'] += 1

        # Xử lý unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id]['age'] += 1

            # Xóa track quá cũ và lưu vào history
            if self.tracks[track_id]['age'] > self.max_age:
                self.history.append({
                    'id': track_id,
                    'center': self.tracks[track_id]['center'],
                    'area': self.tracks[track_id]['area'],
                    'frames_ago': 0
                })
                del self.tracks[track_id]

        # Xử lý unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            center, area = self.get_center_and_area(det)

            # Thử SmartID
            matched_id = self.smart_id_match(center, area)

            if matched_id is not None:
                track_id = matched_id
                # print(f"SmartID: Khôi phục ID {track_id}")
            else:
                track_id = self.next_id
                self.next_id += 1

            self.tracks[track_id] = {
                'bbox': det,
                'center': center,
                'area': area,
                'age': 0,
                'hits': 1
            }

        # Trả về results (chỉ tracks đang active)
        results = [(self.tracks[tid]['bbox'], tid)
                   for tid in self.tracks.keys()]
        # print(self.history)
        return results


path_model_detection = r'C:\Lap trinh\realsense\project\models\best_ad.pt'
model = YOLO(path_model_detection, task='detect')


def visual_tracking(img, tracked_objects):
    for bbox, track_id in tracked_objects:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def tracking_phase(model, rgb, tracker):
    detections = model.predict(rgb, verbose=False, device='cuda:0', conf=0.8, iou=0.3)
    boxes = detections[0].boxes.xyxy.cpu().numpy()
    tracked_objects = tracker.update(boxes)
    sorted_tracked_objects = sorted(tracked_objects, key=lambda x: x[1])
    return sorted_tracked_objects


if __name__ == "__main__":
    pipeline, align = get_image.pre()
    tracker = PalletTracker()
    list_time = []
    while True:
        rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
        # Giả sử có hàm detect_pallets trả về list bbox
        start = time.time()
        tracked_objects = tracking_phase(model, rgb, tracker)
        # print("Tracked objects:", tracked_objects)
        end = time.time()
        print(end - start)
        list_time.append(end - start)
        rgb = visual_tracking(rgb, tracked_objects)
        cv2.imshow('Tracking', rgb)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            print("Average tracking time:", sum(list_time[1:])/len(list_time))
            cv2.destroyAllWindows()
            break
    
    
# class PalletTracker:
#     def __init__(self,
#                  iou_threshold=0.3,
#                  max_age=5,
#                  his_size=50,
#                  center_threshold=50,
#                  area_threshold=0.5,
#                  reset_after_empty=15):  # 🆕 THÊM THAM SỐ

#         self.tracks = {}
#         self.next_id = 1
#         self.his = deque(maxlen=his_size)

#         self.iou_threshold = iou_threshold
#         self.max_age = max_age
#         self.center_threshold = center_threshold
#         self.area_threshold = area_threshold

#         # 🆕 RESET LOGIC
#         self.reset_after_empty = reset_after_empty
#         self.empty_frame_count = 0  # Đếm số frame liên tiếp không có detection

#     def reset(self):
#         """Reset toàn bộ tracker về trạng thái ban đầu"""
#         self.tracks.clear()
#         self.next_id = 1
#         self.his.clear()
#         self.empty_frame_count = 0
#         print("🔄 Tracker đã reset!")

#     def get_center_area(self, bbox):
#         cx = (bbox[0] + bbox[2]) / 2
#         cy = (bbox[1] + bbox[3]) / 2
#         area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
#         return (cx, cy), area

#     def center_distance(self, c1, c2):
#         return np.linalg.norm(np.array(c1) - np.array(c2))

#     def iou(self, b1, b2):
#         x1 = max(b1[0], b2[0])
#         y1 = max(b1[1], b2[1])
#         x2 = min(b1[2], b2[2])
#         y2 = min(b1[3], b2[3])

#         inter = max(0, x2-x1) * max(0, y2-y1)
#         a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
#         a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
#         union = a1 + a2 - inter

#         return inter / union if union > 0 else 0

#     def find_in_his(self, center, area):
#         matched_ids = []

#         for h in self.his:
#             dist = self.center_distance(center, h['center'])
#             area_diff = abs(area - h['area']) / max(area, h['area'])

#             if dist < self.center_threshold and area_diff < self.area_threshold:
#                 matched_ids.append(h['id'])

#         if len(matched_ids) == 0:
#             return None

#         return min(matched_ids)

#     def normalize_id(self, track_id, center, area):
#         his_id = self.find_in_his(center, area)

#         if his_id is None:
#             self.his.append({
#                 'id': track_id,
#                 'center': center,
#                 'area': area
#             })
#             return track_id

#         return his_id

#     def update(self, detections):
#         results = []

#         # 🆕 KIỂM TRA RESET
#         if len(detections) == 0:
#             self.empty_frame_count += 1

#             if self.empty_frame_count >= self.reset_after_empty:
#                 self.reset()
#                 return []
#         else:
#             self.empty_frame_count = 0  # Reset counter khi có detection

#         # Tăng age
#         for tid in list(self.tracks.keys()):
#             self.tracks[tid]['age'] += 1
#             if self.tracks[tid]['age'] > self.max_age:
#                 del self.tracks[tid]

#         for det in detections:
#             center, area = self.get_center_area(det)

#             # Tìm track match bằng IoU
#             best_tid = None
#             best_iou = 0

#             for tid, t in self.tracks.items():
#                 i = self.iou(det, t['bbox'])
#                 if i > best_iou and i >= self.iou_threshold:
#                     best_iou = i
#                     best_tid = tid

#             if best_tid is None:
#                 tid = self.next_id
#                 self.next_id += 1
#             else:
#                 tid = best_tid

#             # Chuẩn hóa ID qua HIS
#             canonical_id = self.normalize_id(tid, center, area)

#             self.tracks[canonical_id] = {
#                 'bbox': det,
#                 'center': center,
#                 'area': area,
#                 'age': 0
#             }

#             results.append((det, canonical_id))
#         return results

