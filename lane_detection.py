"""
Lane Detection Module - Vanishing Point Clustering + Lane Fill
Chỉ chạy 1 lần trên frame đầu tiên, lưu coeffs để dùng lại cho các frame sau.
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

# ===== CẤU HÌNH =====
LANE_MODEL_PATH = "D:/DATN/model/marrking/last_ver9.pt"

EPS_ANGLE = 2.0     # Ngưỡng góc cluster dotted lines (độ)
FILL_ALPHA = 0.10   # Độ trong suốt màu tô làn

LANE_COLORS = [
    (255, 200,   0),
    ( 60, 220,  80),
    ( 60, 140, 255),
    (220,  60, 255),
    ( 60, 230, 220),
    (255,  80, 160),
]

LINE_COLORS = [
    (255,  80,  80),
    ( 80, 255,  80),
    ( 80,  80, 255),
    (255, 255,  80),
    (255,  80, 255),
    ( 80, 255, 255),
]


# ===== DETECTION CLASS =====
class Detection:
    def __init__(self, label, bbox, confidence, mask=None):
        self.label      = label
        self.bbox       = bbox
        self.confidence = confidence
        self.mask       = mask

    @property
    def x_center(self): return (self.bbox[0] + self.bbox[2]) / 2
    @property
    def y_center(self): return (self.bbox[1] + self.bbox[3]) / 2


# ===== FITTING =====
def fit_line(detection):
    """Fit đường x = f(y) bậc 1 từ mask hoặc bbox."""
    mask = detection.mask
    if mask is not None:
        ys, xs = np.where(mask > 0)
        if len(ys) >= 2 and ys.max() > ys.min():
            bins = np.linspace(ys.min(), ys.max(), 21)
            pts = []
            for i in range(20):
                mask_bin = (ys >= bins[i]) & (ys < bins[i+1])
                if mask_bin.sum() > 0:  
                    x_mean = xs[mask_bin].mean() # trung bình x
                    y_mean = ys[mask_bin].mean() # trung bình y
                    pts.append([x_mean, y_mean])                   
            if len(pts) >= 2:
                pts = np.array(pts)
                return np.polyfit(pts[:, 1], pts[:, 0], 1)

    x1, y1, x2, y2 = detection.bbox
    xc = (x1 + x2) / 2
    ys = np.linspace(y1, y2, 10)
    return np.polyfit(ys, np.full_like(ys, xc), 1)


def fit_cluster(detections):
    """Fit đường x = f(y) bậc 1 qua các tâm dotted line trong cluster."""
    if len(detections) < 2:
        return None
    ys = np.array([d.y_center for d in detections])
    xs = np.array([d.x_center for d in detections])
    return np.polyfit(ys, xs, 1)


# ===== VANISHING POINT =====
def estimate_vanishing_point(detections, img_shape):
    height, width = img_shape[:2]
    solids = [d for d in detections if d.label == 'solid-line']
    params = [fit_line(d) for d in solids]
    params = [p for p in params if p is not None]

    intersections = []
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            a1, b1 = params[i]; a2, b2 = params[j]
            if abs(a1 - a2) < 1e-6:
                continue
            yv = (b2 - b1) / (a1 - a2)
            xv = a1 * yv + b1
            if 0 <= xv <= width and -height <= yv <= height // 2:
                intersections.append((xv, yv))

    if not intersections:
        return (width // 2, height // 8)

    vx = int(np.median([p[0] for p in intersections]))
    vy = int(np.median([p[1] for p in intersections]))
    print(f"  Vanishing point: ({vx}, {vy})")
    return (vx, vy)


# ===== CLUSTERING =====
def cluster_dotted(detections, vp, eps_angle=EPS_ANGLE):
    dotted = [d for d in detections if d.label == 'dotted-line']
    if not dotted:
        return {}

    vx, vy = vp
    angles = np.array([
        np.degrees(np.arctan2(vx - d.x_center, -(vy - d.y_center)))
        for d in dotted
    ]).reshape(-1, 1)
    labels = DBSCAN(eps=eps_angle, min_samples=1).fit(angles).labels_

    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl != -1:
            clusters[lbl].append(dotted[i])

    sorted_items = sorted(clusters.values(), key=lambda ds: np.mean([d.x_center for d in ds]))
    clusters = dict(enumerate(sorted_items))
    print(f"  Dotted clusters: {len(clusters)}")
    return clusters


# ===== PARSE YOLO =====
def parse_yolo(results, img_shape):
    detections = []
    for r in results:
        has_masks = hasattr(r, 'masks') and r.masks is not None
        for i, box in enumerate(r.boxes):
            mask = None
            if has_masks and i < len(r.masks):
                m    = r.masks[i].data.cpu().numpy()[0]
                mask = (cv2.resize(m, (img_shape[1], img_shape[0])) > 0.5).astype(np.uint8)
            detections.append(Detection(
                label      = r.names[int(box.cls[0])],
                bbox       = box.xyxy[0].cpu().numpy(),
                confidence = float(box.conf[0]),
                mask       = mask,
            ))
    return detections


# ===== DETECT LANES (chạy 1 lần) =====
def detect_lanes(model, frame_rgb):
    """
    Chạy lane detection trên frame đầu tiên.
    Trả về: (solid_coeffs, dotted_coeffs, all_lines, vp)
    """
    print("  Running lane detection on first frame...")
    results    = model.predict(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), conf=0.1, verbose=False)
    detections = parse_yolo(results, frame_rgb.shape)

    n_solid  = sum(d.label == 'solid-line'  for d in detections)
    n_dotted = sum(d.label == 'dotted-line' for d in detections)
    print(f"  Detections — solid: {n_solid}, dotted: {n_dotted}")

    solids       = [d for d in detections if d.label == 'solid-line']
    solid_coeffs = [fit_line(d) for d in solids]

    vp = estimate_vanishing_point(detections, frame_rgb.shape)

    clusters      = cluster_dotted(detections, vp, EPS_ANGLE)
    dotted_coeffs = {}
    for cid, ds in clusters.items():
        result = fit_cluster(ds)
        if result is not None:
            dotted_coeffs[cid] = result

    # Gom và sắp xếp tất cả đường biên trái → phải tại đáy ảnh
    h = frame_rgb.shape[0]
    all_lines = sorted(
        [{'coeffs': c, 'type': 'solid'} for c in solid_coeffs if c is not None] +
        [{'coeffs': c, 'type': 'dotted'} for c in dotted_coeffs.values()],
        key=lambda l: np.polyval(l['coeffs'], h)
    )

    n_lanes = max(0, len(all_lines) - 1)
    print(f"  Lane boundaries: {len(all_lines)} → {n_lanes} lanes detected")

    return solid_coeffs, dotted_coeffs, all_lines, vp


# ===== DRAW LANES TRÊN FRAME =====
def draw_lanes_on_frame(frame_bgr, solid_coeffs, dotted_coeffs, all_lines, vp,
                         roi_pts=None, alpha=FILL_ALPHA, y_top_ratio=0.2):
    """
    Vẽ lane overlay lên frame BGR bất kỳ dùng coeffs đã lưu.
    Trả về frame đã vẽ lane.
    """
    result  = frame_bgr.copy()
    h, w    = result.shape[:2]
    overlay = np.zeros_like(result, dtype=np.uint8)

    # Tô màu từng làn
    if len(all_lines) >= 2:
        y_top   = max(int(h * y_top_ratio), int(vp[1]) if vp[1] >= 0 else 0)
        y_range = np.linspace(y_top, h - 1, 300)

        for i in range(len(all_lines) - 1):
            # Điều kiện duy nhất: bỏ qua solid → solid
            if all_lines[i]['type'] == 'solid' and all_lines[i+1]['type'] == 'solid':
                continue
            
            xl = np.clip(np.polyval(all_lines[i]['coeffs'],     y_range), 0, w - 1).astype(int)
            xr = np.clip(np.polyval(all_lines[i + 1]['coeffs'], y_range), 0, w - 1).astype(int)
            poly = np.vstack([
                np.array([xl, y_range], dtype=np.int32).T,
                np.array([xr, y_range], dtype=np.int32).T[::-1]
            ])
            # Chuyển BGR cho cv2
            color_rgb = LANE_COLORS[i % len(LANE_COLORS)]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            # cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], color_bgr)

            mid = len(y_range) // 2

        result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)

    # Vẽ solid lines
    y_range = np.linspace(0, h, 200)
    for coeffs in solid_coeffs:
        if coeffs is None:
            continue
        xs    = np.polyval(coeffs, y_range)
        valid = (xs >= 0) & (xs < w)
        if valid.sum() < 2:
            continue
        pts = np.array([xs[valid], y_range[valid]]).T.astype(np.int32)
        cv2.polylines(result, [pts.reshape(-1, 1, 2)], False, (230, 230, 0), 3)

    # Vẽ dotted lines
    for cid, coeffs in dotted_coeffs.items():
        color_rgb = LINE_COLORS[cid % len(LINE_COLORS)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        xs    = np.polyval(coeffs, y_range)
        valid = (xs >= 0) & (xs < w)
        if valid.sum() < 2:
            continue
        pts = np.array([xs[valid], y_range[valid]]).T.astype(np.int32)
        cv2.polylines(result, [pts.reshape(-1, 1, 2)], False, color_bgr, 2)

    # Mask chỉ giữ lane overlay trong ROI
    if roi_pts is not None:
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(roi_pts, np.int32)], 255)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask_3ch == 255, result, frame_bgr)
        
    return result


# ===== XÁC ĐỊNH XE Ở LÀN NÀO =====
def get_lane_of_point(x, y, all_lines):
    """
    Xác định điểm (x, y) nằm ở làn nào dựa vào all_lines.
    Trả về lane_id (1-based) hoặc None nếu nằm ngoài.
    """
    if len(all_lines) < 2:
        return None

    boundaries = [np.polyval(l['coeffs'], y) for l in all_lines]

    for i in range(len(boundaries) - 1):
        x_left  = min(boundaries[i], boundaries[i + 1])
        x_right = max(boundaries[i], boundaries[i + 1])
        if x_left <= x <= x_right:
            return i + 1

    return None

def filter_coeffs_in_roi(all_lines, roi_pts, frame_height, tolerance=50):
    """Chỉ giữ các đường biên làn nằm trong vùng ROI."""
    x_roi_min = min(p[0] for p in roi_pts)
    x_roi_max = max(p[0] for p in roi_pts)
    y_bottom  = frame_height - 1
    return [l for l in all_lines
            if (x_roi_min - tolerance) <= np.polyval(l['coeffs'], y_bottom) <= (x_roi_max + tolerance)]