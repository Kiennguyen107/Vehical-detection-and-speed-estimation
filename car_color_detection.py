"""
Test phát hiện màu xe dùng KMeans dominant color
ROI: vùng nắp capô (phần dưới ảnh)
Dùng trong Jupyter Notebook
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ===== HSV COLOR RANGES =====
COLOR_RANGES = {
    'red':    ([0,   60,  40], [10,  255, 255]),
    'red2':   ([155, 60,  40], [180, 255, 255]),
    'orange': ([11,  60,  60], [20,  255, 255]),
    'yellow': ([21,  60,  80], [35,  255, 255]),
    'green':  ([36,  50,  40], [85,  255, 255]),
    'blue':   ([86,  50,  40], [130, 255, 255]),
    'purple': ([131, 40,  30], [155, 255, 200]),
    'white':  ([0,   0,  180], [180,  30, 255]),
    'silver': ([0,   0,  150], [180,  35, 185]),
    'black':  ([0,   0,    0], [180, 255,  60]),
    'gray':   ([0,   0,   61], [180,  35, 149]),
    # Thêm vào
    'maroon':      ([0,   80,  30], [10,  255, 120]),   # nâu đỏ tối như Hyundai SantaFe
    'maroon2':     ([155, 80,  30], [180, 255, 120]),   # nâu đỏ tối phía wrap-around
    'brown':       ([10,  60,  30], [20,  200, 140]),   # nâu
    'dark_blue':   ([100, 80,  20], [130, 255, 100]),   # xanh đậm/navy
    'dark_green':  ([36,  60,  20], [85,  255,  80]),   # xanh lá đậm
    'champagne':   ([15,  20,  170],[30,  80,  230]),   # vàng champagne/be
    'beige':       ([10,  10,  180],[25,  50,  240]),   # be/kem
    'bronze':      ([10,  60,  80], [20,  200, 160]),   # đồng
    'gold':        ([20,  80,  120],[35,  255, 200]),   # vàng gold
}


def get_dominant_hsv(img_bgr, n_clusters=4, roi_config=None):
    """
    Dùng KMeans tìm dominant color trong vùng ROI.
    Trả về HSV của cluster có nhiều pixel nhất.
    """
    h, w = img_bgr.shape[:2]

    if roi_config is None:
        roi_config = {
            'y_top':  0.60,
            'y_bot':  0.88,
            'x_left': 0.20,
            'x_right':0.80,
        }

    y1 = int(h * roi_config['y_top'])
    y2 = int(h * roi_config['y_bot'])
    x1 = int(w * roi_config['x_left'])
    x2 = int(w * roi_config['x_right'])

    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None, (y1, y2, x1, x2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    if len(pixels) < n_clusters:
        return None, None, (y1, y2, x1, x2)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Lấy cluster có nhiều pixel nhất
    counts   = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[counts.argmax()]

    return dominant, (kmeans.cluster_centers_, counts), (y1, y2, x1, x2)


def hsv_to_color_name(hsv_pixel):
    """Map HSV pixel sang tên màu."""
    h, s, v = float(hsv_pixel[0]), float(hsv_pixel[1]), float(hsv_pixel[2])

    for name, (lo, hi) in COLOR_RANGES.items():
        # Gộp các biến thể về 1 tên
        if lo[0] <= h <= hi[0] and lo[1] <= s <= hi[1] and lo[2] <= v <= hi[2]:
            return name

    return 'unknown'


def detect_car_color(img_bgr, n_clusters=4, roi_config=None, debug=False):
    """
    Phát hiện màu xe dùng KMeans.
    Trả về (tên màu, HSV dominant, cluster info)
    """
    dominant_hsv, cluster_info, roi_coords = get_dominant_hsv(
        img_bgr, n_clusters=n_clusters, roi_config=roi_config
    )

    if dominant_hsv is None:
        return 'unknown', None, None

    color_name = hsv_to_color_name(dominant_hsv)

    # Gộp red2 → red
    if color_name == 'red2':
        color_name = 'red'
    if color_name in ('red2', 'maroon', 'maroon2'):
        color_name = 'red/maroon'
    if color_name in ('dark_blue',):
        color_name = 'blue'
    if color_name in ('dark_green',):
        color_name = 'green'
    if color_name in ('champagne', 'beige', 'bronze'):
        color_name = 'beige/champagne'

    if debug:
        _show_debug(img_bgr, dominant_hsv, cluster_info, roi_coords, color_name)

    return color_name, dominant_hsv, cluster_info


def _show_debug(img_bgr, dominant_hsv, cluster_info, roi_coords, color_name):
    y1, y2, x1, x2 = roi_coords
    centers, counts = cluster_info

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Ảnh gốc + ROI box
    vis = img_bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Anh goc + ROI (xanh la)", fontsize=11)
    axes[0].axis('off')

    # Vùng ROI
    roi = img_bgr[y1:y2, x1:x2]
    axes[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Vung capo phan tich", fontsize=11)
    axes[1].axis('off')

    # Màu từng cluster (convert HSV → RGB để hiển thị)
    n = len(centers)
    cluster_img = np.zeros((80, n * 80, 3), dtype=np.uint8)
    for i, (center, count) in enumerate(zip(centers, counts)):
        hsv_patch = np.full((80, 80, 3), center, dtype=np.uint8).reshape(1, -1, 3)
        rgb_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2RGB).reshape(80, 80, 3)
        cluster_img[:, i*80:(i+1)*80] = rgb_patch

    axes[2].imshow(cluster_img)
    dominant_idx = counts.argmax()
    axes[2].set_title(
        f"KMeans clusters ({n})\nDominant = cluster {dominant_idx} (vien do)",
        fontsize=10
    )
    # Đánh dấu cluster dominant
    axes[2].add_patch(plt.Rectangle(
        (dominant_idx * 80, 0), 80, 80,
        fill=False, edgecolor='red', linewidth=4
    ))
    for i, c in enumerate(counts):
        axes[2].text(i*80 + 40, 75, str(c), ha='center', va='bottom',
                     fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    axes[2].axis('off')

    # Kết quả + dominant HSV
    dominant_bgr = cv2.cvtColor(
        np.array([[dominant_hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    dominant_rgb = dominant_bgr[::-1]
    result_patch = np.full((80, 200, 3), dominant_rgb, dtype=np.uint8)
    axes[3].imshow(result_patch)
    axes[3].set_title(
        f"Dominant color\nHSV=({dominant_hsv[0]:.0f}, {dominant_hsv[1]:.0f}, {dominant_hsv[2]:.0f})\nKet qua: {color_name.upper()}",
        fontsize=11, fontweight='bold'
    )
    axes[3].axis('off')

    plt.suptitle(f"Mau xe: {color_name.upper()}",
                 fontsize=14, fontweight='bold', color='darkblue')
    plt.tight_layout()
    plt.show()


def test_images(image_paths, n_clusters=4, roi_config=None):
    """Test nhiều ảnh trong Jupyter."""
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Khong doc duoc: {path}")
            continue

        color, dominant_hsv, _ = detect_car_color(
            img, n_clusters=n_clusters, roi_config=roi_config, debug=True
        )
        print(f"File        : {path}")
        print(f"Mau xe      : {color.upper()}")
        if dominant_hsv is not None:
            print(f"HSV dominant: H={dominant_hsv[0]:.0f}, S={dominant_hsv[1]:.0f}, V={dominant_hsv[2]:.0f}")
        print("=" * 40)


# Tuỳ chỉnh nếu kết quả chưa đúng:
# - Tăng n_clusters nếu xe có nhiều màu phức tạp
# - Điều chỉnh ROI nếu vùng capô bị cắt sai
N_CLUSTERS = 8
