"""
Speed Estimation Module - YOLO Tracking + Perspective Transform + Save Line
+ Following Distance Check (khoảng cách an toàn giữa 2 xe cùng làn)
Xe đi xuống (top → bottom): yt nhỏ = gần camera = phía SAU
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime
from queue import Queue

# ===== THÔNG SỐ ĐƯỜNG =====
TARGET_WIDTH  = 15
TARGET_HEIGHT = 52.0

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

DEFAULT_ROI = [
    [1179, 714],
    [2532, 714],
    [3890, 2154],
    [-432, 2154],
]

DEFAULT_SPEED_LIMIT = {"min": 0, "max": 60}

# ===== NGƯỠNG KHOẢNG CÁCH AN TOÀN =====
# Tốc độ < 40 km/h  → không kiểm tra
# 40 – 60  km/h     → 35 m
# 60 – 80  km/h     → 55 m
# 80 – 120 km/h     → 120 m
SAFE_DISTANCE_RULES = [
    (40,   60,  35),
    (60,   80,  55),
    (80,  120, 120),
]

def get_safe_distance(speed_kmh):
    if speed_kmh is None or speed_kmh < 40:
        return None
    for s_min, s_max, dist in SAFE_DISTANCE_RULES:
        if s_min <= speed_kmh <= s_max:
            return dist
    if speed_kmh > 120:
        return 120
    return None


# ===== VIEW TRANSFORMER =====
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        transformed = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2).astype(np.float32), self.m
        )
        return transformed.reshape(-1, 2)


# ===== CROP ROI =====
def crop_roi_region(frame, roi_pts):
    xs    = [p[0] for p in roi_pts]
    ys    = [p[1] for p in roi_pts]
    x_min = max(0, min(xs));  x_max = min(frame.shape[1], max(xs))
    y_min = max(0, min(ys));  y_max = min(frame.shape[0], max(ys))
    crop  = frame[y_min:y_max, x_min:x_max]

    # Resize xuống max 1280px cạnh dài trước khi trả về
    h, w  = crop.shape[:2]
    scale = 1280 / max(h, w)
    if scale < 1.0:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0

    return crop, (x_min, y_min, x_max, y_max), scale


# ===== SPEED ESTIMATOR =====
class SpeedEstimator:
    def __init__(self, roi_pts, fps, lane_speed_limits,
                 crops_output_dir="roi_vehicle_crops",
                 plate_queue: Queue = None,
                 color_queue: Queue = None):
        self.roi_pts            = roi_pts
        self.fps                = fps
        self.lane_speed_limits  = lane_speed_limits
        self.coordinates        = defaultdict(lambda: deque(maxlen=int(fps)))
        self.violations         = []        # vi phạm tốc độ
        self.violated_ids       = set()
        self.all_speeds         = []
        self.speed_recorded_ids = set()
        self.frame_count        = 0
        self.plate_queue        = plate_queue
        self.color_queue        = color_queue

        # ===== KHOẢNG CÁCH =====
        self.distance_violations     = [] 
        self.distance_violated_pairs = set()
        self.unsafe_ids_this_frame   = set()
        self.distance_violated_ids   = set()

        self.transformer = ViewTransformer(np.array(roi_pts, dtype=np.float32), TARGET)
        self.polygon     = np.array(roi_pts, np.int32)

        x_min = min(p[0] for p in roi_pts)
        y_min = min(p[1] for p in roi_pts)
        self.roi_polygon_cropped = np.array(
            [[p[0] - x_min, p[1] - y_min] for p in roi_pts], np.int32
        )

        roi_ys = [p[1] for p in roi_pts]
        self.save_line_y = (min(roi_ys) + max(roi_ys)) // 2 + 600
        self.saved_ids   = set()

        self.violation_output_dir = crops_output_dir + "_violations"
        Path(self.violation_output_dir).mkdir(parents=True, exist_ok=True)
        
        self.frame_output_dir = crops_output_dir + "_frames"
        Path(self.frame_output_dir).mkdir(parents=True, exist_ok=True)

    # KIỂM TRA KHOẢNG CÁCH AN TOÀN
    def _check_following_distance(self, lane_positions: dict, current_speeds: dict,
                                frame_original=None, tid_to_box: dict = None):
        self.unsafe_ids_this_frame = set()

        for lane_id, vehicles in lane_positions.items():
            if len(vehicles) < 2:
                continue

            vehicles_sorted = sorted(vehicles, key=lambda v: v[1])

            for idx in range(1, len(vehicles_sorted)):
                tid_behind, yt_behind, yt_behind_top = vehicles_sorted[idx - 1]
                tid_front,  yt_front,  _             = vehicles_sorted[idx]

                distance_m = abs(yt_front - yt_behind_top)
                if distance_m < 24.0:
                    continue

                speed_behind = current_speeds.get(tid_behind)
                safe_dist    = get_safe_distance(speed_behind)

                if safe_dist is None:
                    continue

                if distance_m < safe_dist:
                    self.unsafe_ids_this_frame.add(tid_behind)
                    self.distance_violated_ids.add(tid_behind)

                    pair = (tid_behind, tid_front)
                    if pair not in self.distance_violated_pairs:
                        self.distance_violated_pairs.add(pair)

                        # ===== LƯU FULL FRAME =====
                        frame_path = ''
                        if frame_original is not None and tid_to_box is not None:
                            frame_with_box = frame_original.copy()

                            # Vẽ bbox xe SAU — màu đỏ
                            if tid_behind in tid_to_box:
                                x1b, y1b, x2b, y2b = tid_to_box[tid_behind]
                                cv2.rectangle(frame_with_box,
                                            (x1b, y1b), (x2b, y2b),
                                            (0, 0, 255), 3)
                                label_behind = f"BEHIND #{tid_behind} {int(speed_behind) if speed_behind else '?'}km/h dist:{round(distance_m,1)}m"
                                cv2.putText(frame_with_box, label_behind,
                                            (x1b, y1b - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 0, 255), 2)

                            # Vẽ bbox xe TRƯỚC — màu vàng
                            if tid_front in tid_to_box:
                                x1f, y1f, x2f, y2f = tid_to_box[tid_front]
                                cv2.rectangle(frame_with_box,
                                            (x1f, y1f), (x2f, y2f),
                                            (0, 215, 255), 3)
                                label_front = f"FRONT #{tid_front}"
                                cv2.putText(frame_with_box, label_front,
                                            (x1f, y1f - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 215, 255), 2)

                            # Lưu file
                            lane_str = f"_L{lane_id}" if lane_id else ""
                            speed_str = f"_speed{int(speed_behind)}" if speed_behind else ""
                            frame_path = (
                                f"{self.frame_output_dir}/"
                                f"dist_{tid_behind:04d}_{tid_front:04d}"
                                f"{lane_str}{speed_str}.jpg"
                            )
                            cv2.imwrite(frame_path, frame_with_box)
                        # ==========================

                        self.distance_violations.append({
                            'Time':          datetime.now().strftime("%H:%M:%S"),
                            'Date':          datetime.now().strftime("%Y-%m-%d"),
                            'ID_Behind':     tid_behind,
                            'Distance (m)':  round(distance_m, 1),
                            'Safe_Dist (m)': safe_dist,
                            'Speed (km/h)':  int(speed_behind) if speed_behind else "N/A",
                            'Lane':          f"Làn {lane_id}" if lane_id else "Unknown",
                            'Plate':         'Detecting...',
                            'Frame Path':    frame_path,
                        })

                        for v in self.violations:
                            if v['ID'] == tid_behind:
                                v['Dist Violation'] = True
                                break
                else:
                    self.distance_violated_pairs.discard((tid_behind, tid_front))

    # ------------------------------------------------------------------
    # PROCESS FRAME
    # ------------------------------------------------------------------
    def process_frame(self, frame_with_lanes, frame_original, car_model, all_lines, get_lane_of_point):
        self.frame_count += 1
        cropped, (x_min, y_min, x_max, y_max), scale = crop_roi_region(frame_with_lanes, self.roi_pts)

        result = car_model.track(
            cropped, persist=True, half=False,
            tracker='botsort.yaml', imgsz=640, verbose=False, device="cuda"
        )[0]

        annotated = frame_with_lanes.copy()
        cv2.polylines(annotated, [self.polygon], True, (0, 255, 255), 2)

        n_track = 0

        if not (result.boxes and result.boxes.is_track):
            self._draw_stats(annotated, n_track)
            return annotated, self._stats(n_track)

        boxes       = result.boxes.xywh.cpu()
        track_ids   = result.boxes.id.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes     = result.boxes.cls.int().cpu().tolist()
        class_names = result.names

        points, valid_ids, valid_boxes, valid_confs = [], [], [], []
        for box, tid, conf in zip(boxes, track_ids, confidences):
            x, y, w, h = box
            # Chia scale để đưa tọa độ về crop gốc
            ax = float(x) / scale
            ay = float(y + h / 2) / scale
            if cv2.pointPolygonTest(self.roi_polygon_cropped, (ax, ay), False) >= 0:
                points.append([ax + x_min, ay + y_min])
                valid_ids.append(tid)
                valid_boxes.append(box)
                valid_confs.append(conf)

        n_track = len(valid_ids)
        if not points:
            self._draw_stats(annotated, n_track)
            return annotated, self._stats(n_track)

        transformed = self.transformer.transform_points(np.array(points, dtype=np.float32))

        tid_to_yt = {}
        for tid, (_, yt) in zip(valid_ids, transformed):
            self.coordinates[tid].append(yt)
            tid_to_yt[tid] = yt

        # PASS 1: tính tốc độ + lane
        current_speeds = {}
        tid_to_lane    = {}
        tid_to_box     = {}
        tid_to_conf    = {}
        tid_to_type = {}

        for i, (tid, box) in enumerate(zip(valid_ids, valid_boxes)):
            bx, by, bw, bh = box
            # Chia scale để đưa về tọa độ frame gốc
            x1 = int(bx / scale - bw / scale / 2) + x_min
            y1 = int(by / scale - bh / scale / 2) + y_min
            x2 = int(bx / scale + bw / scale / 2) + x_min
            y2 = int(by / scale + bh / scale / 2) + y_min

            conf  = valid_confs[i]
            ax_g  = float(bx) / scale + x_min
            ay_g  = float(by + bh / 2) / scale + y_min

            lane_id = get_lane_of_point(ax_g, ay_g, all_lines)
            tid_to_lane[tid] = lane_id
            tid_to_box[tid]  = (x1, y1, x2, y2)
            self.tid_to_box_global = tid_to_box
            tid_to_conf[tid] = conf
            tid_to_type[tid] = class_names[classes[i]]

            coords = self.coordinates[tid]
            if len(coords) >= self.fps / 2:
                dist  = abs(coords[-1] - coords[0])
                speed = dist / (len(coords) / self.fps) * 3.6
                current_speeds[tid] = speed
            else:
                current_speeds[tid] = None

        # PASS 2: khoảng cách
        lane_positions = defaultdict(list)
        for tid in valid_ids:
            lane_id = tid_to_lane.get(tid)
            yt      = tid_to_yt.get(tid)
            if lane_id is not None and yt is not None:
                x1, y1, x2, y2 = tid_to_box[tid]
                bx_mid = (x1 + x2) / 2
                yt_top_transformed = self.transformer.transform_points(
                    np.array([[bx_mid, float(y1)]], dtype=np.float32)
                )[0][1]
                lane_positions[lane_id].append((tid, yt, yt_top_transformed))

        self._check_following_distance(lane_positions, current_speeds, frame_original=frame_original, tid_to_box=tid_to_box,)

        # ===== PASS 3: vẽ bbox + xử lý vi phạm tốc độ =====
        for tid in valid_ids:
            x1, y1, x2, y2 = tid_to_box[tid]
            conf     = tid_to_conf[tid]
            lane_id  = tid_to_lane[tid]
            speed    = current_speeds[tid]
            lane_str = f" L{lane_id}" if lane_id else ""
            vehicle_type = tid_to_type.get(tid, '')

            speed_limit = self.lane_speed_limits.get(lane_id, DEFAULT_SPEED_LIMIT)
            violation   = False
            status      = "OK"

            # Màu: cam nếu unsafe following, xanh lá nếu bình thường
            if tid in self.unsafe_ids_this_frame:
                color = (0, 100, 255)
            else:
                color = (0, 255, 0)

            if speed is None:
                label = f"#{tid}{lane_str} [{conf:.2f}]"
            else:
                if tid not in self.speed_recorded_ids:
                    self.speed_recorded_ids.add(tid)
                    self.all_speeds.append({
                        'speed': int(speed),
                        'lane':  f"L{lane_id}" if lane_id else "",
                    })

                if speed < speed_limit["min"]:
                    violation = True
                    status    = "OK"
                    color     = (0, 0, 0)
                    label     = f"#{tid}{lane_str} {vehicle_type} {int(speed)}km/h <{speed_limit['min']}"
                elif speed > speed_limit["max"]:
                    violation = True
                    status    = "TOO_FAST"
                    color     = (0, 0, 255)
                    label     = f"#{tid}{lane_str} {vehicle_type} {int(speed)}km/h >{speed_limit['max']}"
                else:
                    label = f"#{tid}{vehicle_type} {int(speed)}km/h"
                    if tid in self.unsafe_ids_this_frame:
                        label += ""

                if violation and tid not in self.violated_ids:
                    self.violated_ids.add(tid)
                    
                    # Save full frame ngay lúc vi phạm
                    speed_str_  = f"_speed{int(speed)}"
                    lane_str__  = f"_L{lane_id}" if lane_id else ""
                    frame_with_box = frame_original.copy()
                    cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame_with_box, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    frame_path = f"{self.frame_output_dir}/frame_id{tid:04d}{lane_str__}{speed_str_}.jpg"
                    cv2.imwrite(frame_path, frame_with_box)

                    self.violations.append({
                        'Time':           datetime.now().strftime("%H:%M:%S"),
                        'Date':           datetime.now().strftime("%Y-%m-%d"),
                        'ID':             tid,
                        'Speed (km/h)':   int(speed),
                        'Over By (km/h)': int(speed) - speed_limit["max"] + 1,
                        'Lane':           f"L{lane_id}" if lane_id else "Unknown",
                        'Plate':          "Detecting...",
                        'Color':          "Detecting...",
                        'Confidence (%)': round(conf * 100, 1),
                        'Location':       'Highway A1, KM 23',
                        'Status':         status,
                        'Frame Path':     frame_path,
                        'Dist Violation': tid in self.distance_violated_ids,
                    })
                    if tid in self.saved_ids:
                        late_crop = frame_original[y1:(y2 + 100), x1:x2].copy()
                        if late_crop.size > 0:
                            late_path = f"{self.violation_output_dir}/late_id{tid:04d}{lane_str__}{speed_str_}.jpg"
                            cv2.imwrite(late_path, late_crop)
                            if self.plate_queue is not None:
                                self.plate_queue.put({
                                    'tid':  tid,
                                    'path': late_path,
                                })
                            if self.color_queue is not None:
                                self.color_queue.put({
                                    'tid':  tid,
                                    'path': late_path,
                                })

            # ===== SAVE LINE =====
            if tid not in self.saved_ids and (y1 <= self.save_line_y <= y2):
                self.saved_ids.add(tid)
                crop = frame_original[y1:(y2 + 100), x1:x2].copy()
                if crop.size > 0:
                    speed_str = f"_speed{int(speed)}" if speed is not None else ""
                    lane_str_ = f"_L{lane_id}" if lane_id is not None else ""
                    if tid in self.violated_ids and tid in self.distance_violated_ids:
                        prefix = "both"   # vi phạm cả 2
                    elif tid in self.distance_violated_ids:
                        prefix = "dist"   # chỉ vi phạm khoảng cách
                    else:
                        prefix = "speed"  # chỉ vi phạm tốc độ

                    fname = f"{prefix}_id{tid:04d}{lane_str_}{speed_str}.jpg"
                    crop_path = f"{self.violation_output_dir}/{fname}"

                    if tid in self.violated_ids or tid in self.distance_violated_ids:
                        cv2.imwrite(crop_path, crop)
                        if self.plate_queue is not None:
                            self.plate_queue.put({'tid': tid, 'path': crop_path})
                        if self.color_queue is not None:
                            self.color_queue.put({'tid': tid, 'path': crop_path})

            # ===== VẼ BBOX + LABEL =====
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        self._draw_stats(annotated, n_track)
        return annotated, self._stats(n_track)

    # ------------------------------------------------------------------
    def _draw_stats(self, frame, n_track):
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 100), (255, 255, 255), 2)
        cv2.putText(frame, f"Counted Vehicles: {len(self.saved_ids)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracking: {n_track}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, th), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
        x = frame.shape[1] - tw - 20  # góc phải
        cv2.rectangle(frame, (x - 5, 10), (x + tw + 5, th + 20), (0, 0, 0), -1)
        cv2.putText(frame, timestamp, (x, th + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

    def _stats(self, n_track):
        return {
            'vehicle_count':    len(self.saved_ids),
            'current_tracking': n_track,
            'violations_count': len(self.violations),
            'frame_count':      self.frame_count,
            'all_speeds':       self.all_speeds,
        }