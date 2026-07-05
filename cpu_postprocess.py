#!/usr/bin/env python3
"""
NumPy port of RTM3D's geometric solver (gen_position + car_pose_decode).
Runs on ARM CPU after NPU outputs feature maps.
Pure NumPy — no PyTorch dependency on board.

Ported from: RTM3D-master/src/lib/models/decode.py (gen_position, car_pose_decode)
"""

import numpy as np


# ---- Heatmap utilities ----

def _nms(heat, kernel=3):
    """Max-pool NMS on heatmap."""
    from scipy.ndimage import maximum_filter
    hmax = maximum_filter(heat, size=kernel, mode='constant')
    keep = (hmax == heat).astype(np.float32)
    return heat * keep


def _topk(scores, K=40):
    """Find top-K peaks per batch, per class on a 2D heatmap.
    scores: (H, W)
    Returns: scores (K,), indices (K,), ys (K,), xs (K,)
    """
    H, W = scores.shape
    flat = scores.flatten()
    idx = np.argpartition(flat, -K)[-K:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    topk_scores = flat[idx]
    ys = (idx // W).astype(np.float32)
    xs = (idx % W).astype(np.float32)
    return topk_scores, idx, ys, xs


def _topk_channel(scores, K=40):
    """Top-K per channel.
    scores: (C, H, W)
    Returns: scores (C, K), inds (C, K), ys (C, K), xs (C, K)
    """
    C, H, W = scores.shape
    topk_scores = np.zeros((C, K), dtype=np.float32)
    topk_inds = np.zeros((C, K), dtype=np.int64)
    topk_ys = np.zeros((C, K), dtype=np.float32)
    topk_xs = np.zeros((C, K), dtype=np.float32)
    for c in range(C):
        sc, ind, y, x = _topk(scores[c], K)
        topk_scores[c] = sc
        topk_inds[c] = ind
        topk_ys[c] = y
        topk_xs[c] = x
    return topk_scores, topk_inds, topk_ys, topk_xs


# ---- Main decode ----

def car_pose_decode_np(outputs, K=200, conf_thresh=0.15, down_ratio=4):
    """
    Decode NPU outputs into 3D detections.

    Args:
        outputs: dict with keys:
            'hm': (3, H, W) center heatmap
            'wh': (2, H, W) 2D box size
            'hps': (18, H, W) 9 keypoint offsets relative to center
            'dim': (3, H, W) 3D dimensions
            'rot': (8, H, W) rotation encoding
            'prob': (1, H, W) confidence
            'reg': (2, H, W) center offset
            'hm_hp': (9, H, W) keypoint heatmaps
            'hp_offset': (2, H, W) keypoint offsets
        K: max detections
        conf_thresh: confidence threshold
        down_ratio: output stride (4)

    Returns:
        detections: list of dicts with keys:
            class, confidence, bbox_2d, dimensions_3d,
            location_3d, yaw, kps_2d, kps_3d
    """
    hm = outputs['hm']         # (3, H, W)
    wh = outputs['wh']         # (2, H, W)
    hps = outputs['hps']       # (18, H, W)
    dim_out = outputs['dim']   # (3, H, W)
    rot = outputs['rot']       # (8, H, W)
    prob = outputs['prob']     # (1, H, W)
    reg = outputs['reg']       # (2, H, W)
    hm_hp = outputs['hm_hp']   # (9, H, W)
    hp_offset = outputs['hp_offset']  # (2, H, W)
    calib = outputs.get('calib')  # (3, 4) projection matrix, optional

    num_classes, H, W = hm.shape
    num_joints = 9

    # Sigmoid heatmaps (already applied if model has sigmoid, but ensure)
    # hm is typically sigmoid in training loss but raw logits from model

    # Take top-K from RAW heatmap (no HM-level NMS — use box-level NMS later)
    hm_flat = hm.reshape(-1)
    topk_idx = np.argpartition(hm_flat, -K)[-K:]
    topk_idx = topk_idx[np.argsort(hm_flat[topk_idx])[::-1]]

    topk_scores = hm_flat[topk_idx]
    topk_clses = (topk_idx // (H * W)).astype(np.int32)
    topk_inds = topk_idx % (H * W)
    topk_ys = (topk_inds // W).astype(np.float32)
    topk_xs = (topk_inds % W).astype(np.float32)

    # Filter by confidence
    valid = topk_scores > conf_thresh
    if not np.any(valid):
        return []

    topk_scores = topk_scores[valid]
    topk_clses = topk_clses[valid]
    topk_inds = topk_inds[valid]
    topk_ys = topk_ys[valid]
    topk_xs = topk_xs[valid]
    K_actual = len(topk_scores)

    # Gather features at detection centers
    # reg: add center offset
    reg_x = reg[0].flat[topk_inds]
    reg_y = reg[1].flat[topk_inds]
    xs = topk_xs + reg_x
    ys = topk_ys + reg_y

    # wh: 2D box size
    wh_x = wh[0].flat[topk_inds]
    wh_y = wh[1].flat[topk_inds]
    bboxes = np.stack([
        xs - wh_x / 2, ys - wh_y / 2,
        xs + wh_x / 2, ys + wh_y / 2
    ], axis=1)  # (K, 4)

    # Scale to input resolution
    bboxes *= down_ratio
    xs_img = xs * down_ratio
    ys_img = ys * down_ratio

    # dim: 3D dimensions (decode from log-space using DAIR-V2X priors)
    DIM_PRIOR = np.array([1.28, 1.49, 4.37], dtype=np.float32)
    dims = np.zeros((K_actual, 3), dtype=np.float32)
    for i in range(3):
        dims[:, i] = dim_out[i].flat[topk_inds]
    dims = np.exp(dims) * DIM_PRIOR  # decode

    # hps: 9 keypoint offsets (relative to center, in feature map coords)
    kps_2d = np.zeros((K_actual, 9, 2), dtype=np.float32)
    for j in range(9):
        kps_2d[:, j, 0] = hps[j * 2].flat[topk_inds] + topk_xs
        kps_2d[:, j, 1] = hps[j * 2 + 1].flat[topk_inds] + topk_ys
    kps_2d *= down_ratio  # scale to image coords

    # rot: rotation encoding (8 channels)
    rot_vals = np.zeros((K_actual, 8), dtype=np.float32)
    for i in range(8):
        rot_vals[:, i] = rot[i].flat[topk_inds]

    # prob: confidence
    prob_vals = prob[0].flat[topk_inds]

    # ---- Geometric 3D position solver (gen_position port) ----
    # This solves for [X, Y, Z]^T in camera frame using the overdetermined
    # system from 9 keypoints + 3D dimensions + rotation.

    positions = gen_position_np(kps_2d, dims, rot_vals, calib)

    # Build raw results
    raw_results = []
    CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
    for i in range(K_actual):
        cls_id = topk_clses[i]
        if cls_id >= len(CLASS_NAMES):
            continue
        raw_results.append({
            'class': CLASS_NAMES[cls_id],
            'class_id': int(cls_id),
            'confidence': float(topk_scores[i]),
            'bbox_2d': {
                'xmin': float(bboxes[i, 0]), 'ymin': float(bboxes[i, 1]),
                'xmax': float(bboxes[i, 2]), 'ymax': float(bboxes[i, 3]),
            },
            'dimensions_3d': {
                'h': float(dims[i, 0]), 'w': float(dims[i, 1]), 'l': float(dims[i, 2]),
            },
            'location_3d': {
                'x': float(positions[i, 0]), 'y': float(positions[i, 1]), 'z': float(positions[i, 2]),
            },
            'yaw': float(positions[i, 3]),
        })

    # Box-level IoU NMS (soft suppression — keep both if IoU < threshold)
    return _box_nms(raw_results, iou_thresh=0.5)


def _box_nms(detections, iou_thresh=0.5):
    """IoU-based NMS on 2D boxes. Keeps higher-confidence box, suppresses overlapping ones."""
    if len(detections) <= 1:
        return detections
    # Sort by confidence descending
    dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    suppressed = set()
    for i, di in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(di)
        for j, dj in enumerate(dets[i+1:], i+1):
            if j in suppressed:
                continue
            if di['class'] != dj['class']:
                continue  # only suppress same class
            # IOU
            b1 = [di['bbox_2d']['xmin'], di['bbox_2d']['ymin'], di['bbox_2d']['xmax'], di['bbox_2d']['ymax']]
            b2 = [dj['bbox_2d']['xmin'], dj['bbox_2d']['ymin'], dj['bbox_2d']['xmax'], dj['bbox_2d']['ymax']]
            x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
            a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou >= iou_thresh:
                suppressed.add(j)
    return keep


def gen_position_np(kps, dim, rot, calib=None):
    """
    Fully vectorized NumPy port of RTM3D's gen_position().
    Solves the 3D position [X, Y, Z] in batch for all K objects simultaneously.
    """
    K = kps.shape[0]
    if K == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Default calib (identity)
    if calib is None:
        calib = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32)

    fx = calib[0, 0]
    cx = calib[0, 2]
    cy = calib[1, 2]

    # ---- Vectorized Rotation Estimation ----
    alpha_idx = (rot[:, 1] > rot[:, 5]).astype(np.float32)
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) - 0.5 * np.pi
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + 0.5 * np.pi
    alpha_pre = alpha1 * alpha_idx + alpha2 * (1.0 - alpha_idx)

    # rotation_y = alpha + atan2(x_cam - cx, fx)
    rotation_y = alpha_pre + np.arctan2(kps[:, 8, 0] - cx, fx)
    rotation_y = np.clip(rotation_y, -np.pi, np.pi)

    # ---- Batch Corner Coordinates Setup ----
    h, w, l = dim[:, 0], dim[:, 1], dim[:, 2]
    corner_coords = np.zeros((K, 9, 3), dtype=np.float32)
    corner_coords[:, 0, 0] = l / 2;  corner_coords[:, 0, 1] = h / 2;  corner_coords[:, 0, 2] = w / 2
    corner_coords[:, 1, 0] = l / 2;  corner_coords[:, 1, 1] = h / 2;  corner_coords[:, 1, 2] = -w / 2
    corner_coords[:, 2, 0] = -l / 2; corner_coords[:, 2, 1] = h / 2;  corner_coords[:, 2, 2] = -w / 2
    corner_coords[:, 3, 0] = -l / 2; corner_coords[:, 3, 1] = h / 2;  corner_coords[:, 3, 2] = w / 2
    # index 8 (center) is [0, 0, 0]

    # ---- Batch Rotate Visible Corners ----
    VISIBLE_INDICES = [0, 1, 2, 3, 8]
    vis_corners = corner_coords[:, VISIBLE_INDICES, :]  # (K, 5, 3)
    x_c, y_c, z_c = vis_corners[:, :, 0], vis_corners[:, :, 1], vis_corners[:, :, 2]

    cosori = np.cos(rotation_y)[:, np.newaxis]  # (K, 1)
    sinori = np.sin(rotation_y)[:, np.newaxis]  # (K, 1)

    corner_rot_x = x_c * cosori + z_c * sinori  # (K, 5)
    corner_rot_y = y_c                          # (K, 5)
    corner_rot_z = -x_c * sinori + z_c * cosori # (K, 5)

    # ---- Batch Build A and B Matrices ----
    kp_norm = np.zeros((K, 9, 2), dtype=np.float32)
    kp_norm[:, :, 0] = (kps[:, :, 0] - cx) / fx
    kp_norm[:, :, 1] = (kps[:, :, 1] - cy) / fx

    kp_norm_vis_x = kp_norm[:, VISIBLE_INDICES, 0]  # (K, 5)
    kp_norm_vis_y = kp_norm[:, VISIBLE_INDICES, 1]  # (K, 5)

    A = np.zeros((K, 10, 3), dtype=np.float32)
    A[:, 0::2, 0] = -1
    A[:, 1::2, 1] = -1
    A[:, 0::2, 2] = kp_norm_vis_x
    A[:, 1::2, 2] = kp_norm_vis_y

    B = np.zeros((K, 10, 1), dtype=np.float32)
    B[:, 0::2, 0] = corner_rot_x - kp_norm_vis_x * corner_rot_z
    B[:, 1::2, 0] = corner_rot_y - kp_norm_vis_y * corner_rot_z

    # ---- Batch Solve: A^T A X = A^T B ----
    AT = A.transpose(0, 2, 1)  # (K, 3, 10)
    ATA = np.matmul(AT, A)      # (K, 3, 3)
    ATB = np.matmul(AT, B)      # (K, 3, 1)

    # Add a small regularizer to guarantee invertibility (1e-6)
    reg = 1e-6 * np.eye(3, dtype=np.float32)[np.newaxis, :, :]
    X = np.linalg.solve(ATA + reg, ATB)  # (K, 3, 1)

    X_pos = X[:, :, 0]  # (K, 3)
    # Adjust Y to be the center (RTM3D outputs bbox bottom Y)
    X_pos[:, 1] += h / 2.0

    results = np.zeros((K, 4), dtype=np.float32)
    results[:, :3] = X_pos
    results[:, 3] = rotation_y

    return results


def _get_corner(j, h, w, l):
    """Get 3D bbox corner coordinates in object frame.
    j: vertex index (0-8, where 8 is center)
    h, w, l: height, width, length
    Returns: (x, y, z) in object frame
    """
    corners = [
        [ l/2,  h/2,  w/2],  # 0
        [ l/2,  h/2, -w/2],  # 1
        [-l/2,  h/2, -w/2],  # 2
        [-l/2,  h/2,  w/2],  # 3
        [ l/2, -h/2,  w/2],  # 4
        [ l/2, -h/2, -w/2],  # 5
        [-l/2, -h/2, -w/2],  # 6
        [-l/2, -h/2,  w/2],  # 7
        [ 0.0,  0.0,  0.0],  # 8 center
    ]
    return corners[j]


# ---- Test ----

if __name__ == '__main__':
    # Generate mock NPU outputs matching rtm3d_resnet18.onnx
    H, W = 96, 320
    np.random.seed(42)

    # Create a mock detection: car at (40, 160) in feature map
    mock_outputs = {
        'hm': np.zeros((3, H, W), dtype=np.float32),
        'wh': np.zeros((2, H, W), dtype=np.float32),
        'hps': np.zeros((18, H, W), dtype=np.float32),
        'dim': np.zeros((3, H, W), dtype=np.float32),
        'rot': np.zeros((8, H, W), dtype=np.float32),
        'prob': np.zeros((1, H, W), dtype=np.float32),
        'reg': np.zeros((2, H, W), dtype=np.float32),
        'hm_hp': np.zeros((9, H, W), dtype=np.float32),
        'hp_offset': np.zeros((2, H, W), dtype=np.float32),
        'calib': np.array([[2183.375, 0, 940.590, 0],
                           [0, 2329.297, 567.568, 0],
                           [0, 0, 1, 0]], dtype=np.float32),
    }

    # Place a strong car detection at center of feature map position (48, 160)
    cy, cx = 48, 160
    mock_outputs['hm'][0, cy-2:cy+3, cx-2:cx+3] = 0.9
    mock_outputs['hm'][0, cy, cx] = 0.95
    mock_outputs['wh'][0, cy, cx] = 50 / 4   # ~50px width at stride 4
    mock_outputs['wh'][1, cy, cx] = 40 / 4   # ~40px height
    mock_outputs['dim'][0, cy, cx] = 0.0  # h offset
    mock_outputs['dim'][1, cy, cx] = 0.0  # w offset
    mock_outputs['dim'][2, cy, cx] = 0.0  # l offset
    mock_outputs['rot'][1, cy, cx] = 0.5  # bin1_alpha
    mock_outputs['rot'][4, cy, cx] = 0.3  # bin2
    mock_outputs['rot'][5, cy, cx] = -0.2
    mock_outputs['prob'][0, cy, cx] = 0.8

    # Set keypoint offsets (small values - at corners of the detection box)
    for j in range(9):
        mock_outputs['hps'][j*2, cy, cx] = 0
        mock_outputs['hps'][j*2+1, cy, cx] = 0

    # Set hm_hp heatmaps
    for j in range(9):
        mock_outputs['hm_hp'][j, cy, cx] = 0.5

    # Decode
    detections = car_pose_decode_np(mock_outputs, K=10, conf_thresh=0.3)
    print(f'Mock test: {len(detections)} detections')
    for d in detections:
        print(f'  {d["class"]} conf={d["confidence"]:.3f} '
              f'pos=({d["location_3d"]["x"]:.1f}, {d["location_3d"]["y"]:.1f}, {d["location_3d"]["z"]:.1f}) '
              f'yaw={d["yaw"]:.2f}')
