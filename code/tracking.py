import numpy as np
from collections import deque
import time
import get_image
import cv2
from ultralytics import YOLO
import time


class PalletTracker:
    def __init__(self, iou_threshold=0.3, max_age=1, history_size=10,
                 smart_center_threshold=300, smart_area_threshold=0.6,
                 history_ttl=15, use_hungarian=False):
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

    def cleanup_history(self):
        """
        Remove expired history entries
        """
        self.history = deque(
            [h for h in self.history if h['frames_ago'] <= self.history_ttl],
            maxlen=self.history.maxlen
        )
        
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
            if self.empty_frames >= 30:
                print(f"Reset tracker sau {self.empty_frames} frames rỗng")
                self.reset()

            # Tăng frames_ago cho history
            for hist in self.history:
                hist['frames_ago'] += 1

            self.cleanup_history()
            return []

        # Reset empty counter
        self.empty_frames = 0

        # Tăng frames_ago cho history
        for hist in self.history:
            hist['frames_ago'] += 1
        
        self.cleanup_history()

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

path_model_detection = r"C:\Lap trinh\realsense\final_project\model\best.pt"

def load_mode(path_model_detection):
    model = YOLO(path_model_detection, task='detect')

    return model
def visual_tracking(img, tracked_objects):
    for bbox, track_id in tracked_objects:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def tracking_phase(model, rgb, tracker):
    start_predict = time.time()
    detections = model.predict(rgb, verbose=False, device='cuda:0', conf=0.85)
    end_predict = time.time()
    # print("Time for predict: ", end_predict - start_predict)
    boxes_after = []
    boxes = detections[0].boxes.xyxy.cpu().numpy()
    for box in boxes:
        if((box[2] - box[0]) > (3*(box[3] - box[1]))):
            boxes_after.append(box)
    print('box', type(boxes_after))
    start_update = time.time()
    
    tracked_objects = tracker.update(boxes_after)
    end_update = time.time()
    # print("Time for updateing: ", end_update - start_update)
    sorted_tracked_objects = sorted(tracked_objects, key=lambda x: x[1])
    return sorted_tracked_objects


# def merge_split_objects_by_center(detections, track_ids=None,
#                                   center_distance_threshold=300,
#                                   max_individual_ar=7.0):
#     """
#     Merge boxes bị chia bởi vật cản
    
#     Logic:
#     - Chỉ check AR của TỪNG box riêng lẻ (tránh merge 2 pallet)
#     - KHÔNG check AR của merged box (vì merge 2 phần lại mới thành object hoàn chỉnh)
#     """
#     if len(detections) == 0:
#         return [], track_ids if track_ids is not None else None

#     if isinstance(detections, list):
#         detections = np.array(detections)

#     # Tính centers và AR
#     boxes_info = []
#     for i, det in enumerate(detections):
#         if len(det) >= 6:
#             x1, y1, x2, y2, conf, cls = det[:6]
#         else:
#             x1, y1, x2, y2 = det[:4]
#             conf, cls = 1.0, 0

#         cx = (x1 + x2) / 2
#         cy = (y1 + y2) / 2
#         w = x2 - x1
#         h = y2 - y1
#         ar = max(w, h) / (min(w, h) + 1e-6)

#         boxes_info.append({
#             'bbox': [x1, y1, x2, y2],
#             'center': (cx, cy),
#             'ar': ar,
#             'cls': cls,
#             'idx': i
#         })

#     # Union-Find
#     parent = list(range(len(detections)))

#     def find(x):
#         if parent[x] != x:
#             parent[x] = find(parent[x])
#         return parent[x]

#     def union(x, y):
#         px, py = find(x), find(y)
#         if px != py:
#             parent[px] = py

#     # Merge logic
#     for i in range(len(boxes_info)):
#         box_i = boxes_info[i]

#         # Nếu box i là pallet nguyên (AR cao) → SKIP
#         if box_i['ar'] >= max_individual_ar:
#             continue

#         for j in range(i + 1, len(boxes_info)):
#             box_j = boxes_info[j]

#             # Nếu box j là pallet nguyên → SKIP
#             if box_j['ar'] >= max_individual_ar:
#                 continue

#             # Cùng class
#             if box_i['cls'] != box_j['cls']:
#                 continue

#             # Tính distance
#             cx1, cy1 = box_i['center']
#             cx2, cy2 = box_j['center']
#             distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

#             # Merge nếu tâm gần
#             # KHÔNG check merged AR nữa!
#             if distance < center_distance_threshold:
#                 union(i, j)

#     # Group boxes
#     groups = {}
#     for i in range(len(boxes_info)):
#         root = find(i)
#         if root not in groups:
#             groups[root] = []
#         groups[root].append(i)

#     # Merge từng group
#     merged = []
#     merged_ids = []

#     for root, indices in groups.items():
#         group_boxes = [boxes_info[idx]['bbox'] for idx in indices]

#         xs1 = [b[0] for b in group_boxes]
#         ys1 = [b[1] for b in group_boxes]
#         xs2 = [b[2] for b in group_boxes]
#         ys2 = [b[3] for b in group_boxes]

#         merged_box = np.array([
#             min(xs1),
#             min(ys1),
#             max(xs2),
#             max(ys2)
#         ], dtype=np.float32)

#         merged.append(merged_box)

#         if track_ids is not None:
#             group_track_ids = [track_ids[idx] for idx in indices]
#             merged_ids.append(min(group_track_ids))

#     merged_ids = np.array(merged_ids) if track_ids is not None and len(
#         merged_ids) > 0 else None

#     return merged, merged_ids


def merge_split_objects_by_center(detections, track_ids=None,
                                  center_distance_threshold=300,
                                  max_individual_ar=4):
    if len(detections) == 0:
        return [], track_ids if track_ids is not None else None

    if isinstance(detections, list):
        detections = np.array(detections)

    # print(f"\n=== MERGE START ===")
    # print(f"Input: {len(detections)} boxes")
    # print(
    #     f"Thresholds: distance={center_distance_threshold}, AR={max_individual_ar}")
    
    # Tính centers và AR
    boxes_info = []
    for i, det in enumerate(detections):
        if len(det) >= 6:
            x1, y1, x2, y2, conf, cls = det[:6]
        else:
            x1, y1, x2, y2 = det[:4]
            conf, cls = 1.0, 0

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        ar = max(w, h) / (min(w, h) + 1e-6)

        boxes_info.append({
            'bbox': [x1, y1, x2, y2],
            'center': (cx, cy),
            'ar': ar,
            'cls': cls,
            'idx': i
        })

        # print(
        #     f"Box {i}: bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], center=({cx:.1f},{cy:.1f}), AR={ar:.2f}, cls={cls}")

    # Union-Find
    parent = list(range(len(detections)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            # print(f"✅ MERGED: Box {x} + Box {y}")

    # Merge logic
    merge_attempts = 0
    for i in range(len(boxes_info)):
        box_i = boxes_info[i]

        # Box i là pallet nguyên?
        if box_i['ar'] >= max_individual_ar:
            # print(
            #     f"❌ Box {i}: AR={box_i['ar']:.2f} >= {max_individual_ar} → SKIP (pallet nguyên)")
            continue

        for j in range(i + 1, len(boxes_info)):
            box_j = boxes_info[j]

            merge_attempts += 1
            # print(f"\n--- Checking Box {i} + Box {j} ---")

            # Box j là pallet nguyên?
            if box_j['ar'] >= max_individual_ar:
                # print(
                #     f"❌ Box {j}: AR={box_j['ar']:.2f} >= {max_individual_ar} → SKIP")
                continue

            # Cùng class?
            if box_i['cls'] != box_j['cls']:
                # print(f"❌ Different class: {box_i['cls']} vs {box_j['cls']}")
                continue

            # Tính distance
            cx1, cy1 = box_i['center']
            cx2, cy2 = box_j['center']
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

            # print(
            #     f"Distance: {distance:.1f}px (threshold: {center_distance_threshold})")

            # Merge?
            if distance < center_distance_threshold:
                # print(f"✅ Distance OK → MERGING!")
                union(i, j)

    # print(f"\nTotal merge attempts: {merge_attempts}")

    # Group boxes
    groups = {}
    for i in range(len(boxes_info)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # print(f"\nGroups formed: {len(groups)}")
    # for root, indices in groups.items():
    #     print(f"  Group {root}: boxes {indices}")

    # Merge từng group
    merged = []
    merged_ids = []

    for root, indices in groups.items():
        group_boxes = [boxes_info[idx]['bbox'] for idx in indices]

        xs1 = [b[0] for b in group_boxes]
        ys1 = [b[1] for b in group_boxes]
        xs2 = [b[2] for b in group_boxes]
        ys2 = [b[3] for b in group_boxes]

        merged_box = np.array([
            min(xs1),
            min(ys1),
            max(xs2),
            max(ys2)
        ], dtype=np.float32)

        merged.append(merged_box)

        # print(
        #     f"  → Merged: [{min(xs1):.0f},{min(ys1):.0f},{max(xs2):.0f},{max(ys2):.0f}]")

        if track_ids is not None:
            group_track_ids = [track_ids[idx] for idx in indices]
            merged_ids.append(min(group_track_ids))

    # print(f"Output: {len(merged)} boxes")
    # print(f"===================\n")

    merged_ids = np.array(merged_ids) if track_ids is not None and len(
        merged_ids) > 0 else None

    return merged, merged_ids

def tracking_phase_ad(model, rgb, tracker):
    """
    Pipeline tracking: Detection → Merge → Update Tracker
    """
    # YOLO Detection
    results = model.predict(rgb, conf=0.8, verbose=False)

    # Lấy detections
    boxes = results[0].boxes
    detections = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()
        detections.append([x1, y1, x2, y2, conf, cls])

    
    detections = np.array(detections)
        # Merge các boxes bị chia bởi vật cản
    merged_detections, _ = merge_split_objects_by_center(
        detections,
        center_distance_threshold=300,  # Khoảng cách tâm tối đa
        max_individual_ar=4    # AR < 5 mới merge
    )

    # Tracking với merged boxes8       
    tracked_objects = tracker.update(merged_detections)
        
    # Sắp xếp theo ID
    sorted_tracked_objects = sorted(tracked_objects, key=lambda x: x[1])

    return sorted_tracked_objects

if __name__ == "__main__":
    pipeline, align = get_image.pre()
    tracker = PalletTracker()
    list_time = []
    count = 0
    model = load_mode(path_model_detection)
    while True:
        rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
        # Giả sử có hàm detect_pallets trả về list bbox
        start = time.time()
        tracked_objects = tracking_phase_ad(model, rgb, tracker)
        # print(tracked_objects)
        end = time.time()
        # print(end - start)
        list_time.append(end - start)
        rgb = visual_tracking(rgb, tracked_objects)
        cv2.imshow('Tracking', rgb)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            print("Average tracking time:", sum(list_time[1:])/len(list_time))
            cv2.destroyAllWindows()
            break
