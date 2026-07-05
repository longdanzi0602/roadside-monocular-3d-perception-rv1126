#!/usr/bin/env python3
"""竞赛展示: 3D检测+热力图二联 + BEV + 统计 + 流量趋势."""

import numpy as np
import cv2
import math

# ── Color Palette ──────────────────────────────────────────────
BG      = (248, 249, 251)
CARD_BG = (255, 255, 255)
BORDER  = (225, 228, 235)
TEXT_1  = (28,  32,  42)
TEXT_2  = (130, 138, 152)
TEXT_3  = (160, 168, 182)
ACCENT  = (59,  130, 246)
GREEN   = (34,  197, 94)
RED     = (239, 68,  68)
GRID    = (238, 241, 246)

CLASS_C = {
    'Car':        (74,  222, 128),  # Neon Mint Green
    'Pedestrian': (255, 191, 0),    # Neon Electric Cyan/Blue
    'Cyclist':    (0,   165, 255),  # Neon Orange-Gold
}
DIM_C = {
    'Car':        (219, 234, 254),
    'Pedestrian': (254, 243, 232),
    'Cyclist':    (237, 233, 254),
}


# ── 3D Box Helpers ────────────────────────────────────────────
def _box_corners_cam(loc, dim, yaw):
    h, w, l = dim['h'], dim['w'], dim['l']
    cs, sn = math.cos(yaw), math.sin(yaw)
    corners = np.array([
        [ l/2,  h/2,  w/2], [ l/2,  h/2, -w/2], [-l/2,  h/2, -w/2], [-l/2,  h/2,  w/2],
        [ l/2, -h/2,  w/2], [ l/2, -h/2, -w/2], [-l/2, -h/2, -w/2], [-l/2, -h/2,  w/2],
    ], dtype=np.float32)
    R = np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]], dtype=np.float32)
    c = R @ corners.T
    center = np.array([loc['x'], loc['y'] - h / 2, loc['z']])
    return c.T + center


def _project(pts, K):
    N = len(pts)
    uv = np.zeros((N, 2))
    ok = np.zeros(N, bool)
    for i in range(N):
        z = pts[i, 2]
        if z > 0.5:
            uv[i] = [K[0, 0] * pts[i, 0] / z + K[0, 2],
                     K[1, 1] * pts[i, 1] / z + K[1, 2]]
            ok[i] = True
    return uv, ok


EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


# ═══════════════════════════════════════════════════════════════
# BEV Bird's-Eye View
# ═══════════════════════════════════════════════════════════════
class BEVRenderer:
    def __init__(self, W=720, H=320, x_range=(-35, 35), z_range=(0, 80)):
        self.W, self.H = W, H
        self.x_range, self.z_range = x_range, z_range
        self.sx = W / (x_range[1] - x_range[0])
        self.sz = H / (z_range[1] - z_range[0])

    def _p(self, x, z):
        return (int((x - self.x_range[0]) * self.sx),
                int((self.z_range[1] - z) * self.sz))

    def render(self, tracks):
        canvas = np.full((self.H, self.W, 3), (248, 250, 252), dtype=np.uint8)
        for d in range(0, int(self.z_range[1]) + 1, 20):
            _, py = self._p(0, d)
            if 0 <= py < self.H:
                cv2.line(canvas, (0, py), (self.W, py), GRID, 1)
                cv2.putText(canvas, f'{d}m', (6, py - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_3, 1)
        cx = self._p(0, 0)[0]
        cv2.line(canvas, (cx, 0), (cx, self.H), (200, 205, 215), 1, cv2.LINE_AA)
        cam_y = self.H - 14
        cv2.drawMarker(canvas, (cx, cam_y), ACCENT, cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.putText(canvas, 'CAM', (cx + 8, cam_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, ACCENT, 1)

        for t in tracks:
            loc = t['location_3d']
            dim = t.get('dimensions_3d', {})
            cls = t['class']
            yaw = t.get('yaw', 0)
            px, py = self._p(loc['x'], loc['z'])
            if not (0 <= px < self.W and 0 <= py < self.H):
                continue
            color = CLASS_C.get(cls, (100, 150, 100))
            # Draw a premium radar-style dot
            cv2.circle(canvas, (px, py), 9, color, 1, cv2.LINE_AA)  # Outer halo
            cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)  # Solid center dot

        cv2.putText(canvas, 'BEV  Top-Down View', (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_2, 1)
        legend_y = self.H - 16
        items = list(CLASS_C.items())
        legend_parts = []
        for name, c in items:
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            legend_parts.append((name, c, tw))
        total_w = sum(10 + tw + 20 for _, _, tw in legend_parts)
        lx = max(8, (self.W - total_w) // 2)
        for name, c, tw in legend_parts:
            cv2.circle(canvas, (lx, legend_y - 2), 5, c, -1)
            cv2.putText(canvas, name, (lx + 10, legend_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_1, 1)
            lx += 10 + tw + 20

        cv2.rectangle(canvas, (0, 0), (self.W - 1, self.H - 1), BORDER, 1)
        return canvas


# ═══════════════════════════════════════════════════════════════
# 3D Detection Panel (left half of detection row)
# ═══════════════════════════════════════════════════════════════
def draw_det_panel(img, tracks, W, H, calib_K=None):
    vis = cv2.resize(img, (W, H))
    h_img, w_img = img.shape[:2]
    sx, sy = W / w_img, H / h_img
    n3d = 0

    for t in tracks:
        cls = t['class']
        color = CLASS_C.get(cls, (100, 150, 100))
        bb = t['bbox_2d']
        x1 = int(np.clip(bb['xmin'] * sx, 0, W))
        y1 = int(np.clip(bb['ymin'] * sy, 0, H))
        x2 = int(np.clip(bb['xmax'] * sx, 0, W))
        y2 = int(np.clip(bb['ymax'] * sy, 0, H))

        # 2D Box: slate grey, thinner, subtle guide
        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 160, 140), 1, cv2.LINE_AA)
        conf = t['confidence']
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.rectangle(vis, (x1, max(y1 - th - 6, 0)),
                      (x1 + tw + 6, y1), (180, 160, 140), -1)
        cv2.putText(vis, label, (x1 + 3, max(y1 - 3, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        # 3D wireframe + Semi-transparent solid face overlay
        loc = t['location_3d']; y3 = loc.get('y', 0)
        dim = t.get('dimensions_3d', {})
        d = math.sqrt(loc['x']**2 + loc['y']**2 + loc['z']**2)
        if calib_K is not None and conf > 0.4 and 3 < d < 200 and abs(y3) < 30 and cls == 'Car':
            try:
                cs = _box_corners_cam(loc, dim, t.get('yaw', 0))
                pts, ok = _project(cs, calib_K)
                if ok.sum() >= 6:
                    pts[:, 0] *= sx; pts[:, 1] *= sy
                    
                    # 1. Draw translucent face overlays to emphasize 3D solidity
                    overlay = vis.copy()
                    # Top face fill
                    if all(ok[idx] for idx in [4, 5, 6, 7]):
                        pts_top = np.array([pts[4], pts[5], pts[6], pts[7]], dtype=np.int32)
                        cv2.fillPoly(overlay, [pts_top], color)
                    # Front face heading highlight (coral/orange)
                    if all(ok[idx] for idx in [0, 1, 5, 4]):
                        pts_front = np.array([pts[0], pts[1], pts[5], pts[4]], dtype=np.int32)
                        cv2.fillPoly(overlay, [pts_front], (59, 130, 246)) # Coral/Red heading
                    cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)
                    
                    # 2. Draw 3D wireframe edges
                    for i, j in EDGES:
                        if ok[i] and ok[j]:
                            # Front face edges: highlight thicker (2px) and distinct color (coral)
                            is_front = (i in [0, 1, 5, 4]) and (j in [0, 1, 5, 4])
                            edge_color = (59, 130, 246) if is_front else color
                            edge_w = 2 if is_front else 1
                            cv2.line(vis, tuple(pts[i].astype(int)),
                                     tuple(pts[j].astype(int)), edge_color, edge_w, cv2.LINE_AA)
                    n3d += 1
            except Exception: pass

    info = f'3D Detection: {len(tracks)} obj  |  {n3d} boxes'
    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(vis, (6, 4), (tw + 16, th + 10), (0, 0, 0, 160), -1)
    cv2.putText(vis, info, (11, th + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
    cv2.rectangle(vis, (0, 0), (W - 1, H - 1), BORDER, 1)
    return vis


# ═══════════════════════════════════════════════════════════════
# Heatmap Panel (right half of detection row)
# ═══════════════════════════════════════════════════════════════
def draw_heatmap_panel(heatmap, W, H):
    """Standalone heatmap panel with JET colormap and scale bar."""
    panel = np.full((H, W, 3), CARD_BG, dtype=np.uint8)

    if heatmap is not None and heatmap.shape[0] >= 1:
        # Sum all class channels for combined heatmap
        hm_sum = heatmap[0].copy()  # Car channel as primary
        if heatmap.shape[0] > 1:
            hm_sum = np.maximum(hm_sum, heatmap[1])  # max with Pedestrian
        hm_sum = np.clip(hm_sum, 0, 1)

        # Resize to panel
        hm_resized = cv2.resize(hm_sum, (W - 20, H - 42), interpolation=cv2.INTER_LINEAR)
        hm_color = cv2.applyColorMap((hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Place in panel
        ox, oy = 10, 28
        panel[oy:oy+hm_color.shape[0], ox:ox+hm_color.shape[1]] = hm_color

        # Scale bar at bottom
        bar_y = oy + hm_color.shape[0] + 6
        bar_x0, bar_x1 = ox, ox + hm_color.shape[1]
        bar_h = 8
        for bx in range(bar_x0, bar_x1):
            t = (bx - bar_x0) / (bar_x1 - bar_x0)
            # JET colormap: blue(0) -> cyan -> green -> yellow -> red(1)
            r = int(np.clip((t - 0.5) * 510, 0, 255))
            g = int(np.clip((0.5 - abs(t - 0.5)) * 510, 0, 255))
            b = int(np.clip((0.5 - t) * 510, 0, 255))
            cv2.line(panel, (bx, bar_y), (bx, bar_y + bar_h), (b, g, r), 1)
        cv2.putText(panel, '0', (bar_x0 - 8, bar_y + bar_h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, TEXT_2, 1)
        cv2.putText(panel, '1', (bar_x1 - 8, bar_y + bar_h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, TEXT_2, 1)
    else:
        cv2.putText(panel, '(heatmap not available)', (W//2 - 80, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_3, 1)

    # Title
    cv2.putText(panel, 'NPU Activation Heatmap', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_2, 1)

    cv2.rectangle(panel, (0, 0), (W - 1, H - 1), BORDER, 1)
    return panel


# ═══════════════════════════════════════════════════════════════
# Stats Panel (right top)
# ═══════════════════════════════════════════════════════════════
def draw_stats(W, H, analysis, timing, quality):
    """Compact stats card fitting H=180px."""
    panel = np.full((H, W, 3), CARD_BG, dtype=np.uint8)
    fps = timing.get('fps', 0)

    fps_str = f'{fps:.1f}'
    (tw, th), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
    cv2.putText(panel, fps_str, (18, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, ACCENT, 3)
    cv2.putText(panel, 'FPS', (24 + tw, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_2, 1)
    total_ms = timing.get('total_ms', 0)
    npu_ms = timing.get('npu_ms', 0)
    post_ms = timing.get('post_ms', 0)
    cv2.putText(panel, f'NPU {npu_ms:.0f}ms  |  Post {post_ms:.0f}ms  |  Total {total_ms:.0f}ms',
                (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_3, 1)

    qx = W - 155
    bright = quality.get('brightness', 0)
    contrast = quality.get('contrast', 0)
    is_dark = quality.get('is_dark', False)
    pill_color = RED if is_dark else GREEN
    pill_text = 'LOW LIGHT' if is_dark else 'NORMAL'
    cv2.circle(panel, (qx, 26), 5, pill_color, -1)
    cv2.putText(panel, pill_text, (qx + 12, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, pill_color, 1)
    cv2.putText(panel, f'Bright {bright:.0f}   Contrast {contrast:.0f}',
                (qx, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_2, 1)

    rt = analysis.get('runtime_sec', 0)
    m, s = int(rt // 60), int(rt % 60)
    cv2.putText(panel, f'Run {m}m{s}s  Frm {analysis.get("total_frames",0)}  Act {analysis.get("active_targets",0)}',
                (qx, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_3, 1)

    cv2.line(panel, (18, 88), (W - 18, 88), GRID, 1)

    ct = analysis.get('class_total', {})
    total = max(sum(ct.values()), 1)

    for i, name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
        pct = ct.get(name, 0) / total * 100
        color = CLASS_C.get(name, (100, 100, 100))
        count = ct.get(name, 0)
        lx = 18 + i * 150
        cv2.circle(panel, (lx, 108), 5, color, -1)
        cv2.putText(panel, f'{name} {count}', (lx + 12, 113),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
        bar_y = 122
        cv2.rectangle(panel, (lx, bar_y), (lx + 140, bar_y + 14), GRID, -1)
        fill_w = int(140 * pct / 100)
        if fill_w > 0:
            cv2.rectangle(panel, (lx, bar_y), (lx + fill_w, bar_y + 14), color, -1)
        cv2.putText(panel, f'{pct:.0f}%', (lx + fill_w + 4, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    cv2.rectangle(panel, (0, 0), (W - 1, H - 1), BORDER, 1)
    return panel


# ═══════════════════════════════════════════════════════════════
# Flow Trend Chart (right bottom) — flow bars only
# ═══════════════════════════════════════════════════════════════
def draw_trend(W, H, flow):
    """Bar chart: vehicles per frame over time."""
    panel = np.full((H, W, 3), CARD_BG, dtype=np.uint8)

    ml, mr, mt, mb = 52, 24, 34, 32
    cw = W - ml - mr
    ch = H - mt - mb
    cx0, cy0 = ml, mt
    cx1, cy1 = ml + cw, mt + ch

    if cw < 10 or ch < 10:
        return panel

    cv2.putText(panel, 'Traffic Flow', (14, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_1, 1)

    cv2.rectangle(panel, (cx0, cy0), (cx1, cy1), (250, 251, 254), -1)
    cv2.rectangle(panel, (cx0, cy0), (cx1, cy1), GRID, 1)

    has_flow = len(flow) > 1
    max_flow = max(flow) if has_flow else 1
    y_flow_max = max(5, int(max_flow * 1.3 + 1))
    y_fstep = max(1, y_flow_max // 4)

    for v in range(0, y_flow_max + 1, y_fstep):
        yy = cy1 - int(v / y_flow_max * ch) if y_flow_max > 0 else cy1
        if cx0 <= yy <= cy1:
            cv2.line(panel, (cx0, yy), (cx1, yy), GRID, 1)
        cv2.putText(panel, str(v), (cx0 - 34, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, GREEN, 1)
    cv2.putText(panel, 'veh/frame', (cx0 - 34, cy0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, GREEN, 1)

    if has_flow:
        n_bars = len(flow)
        bar_w = max(2, (cw - 6) // n_bars - 1)
        gap = max(1, (cw - 6) // n_bars - bar_w)
        for i, v in enumerate(flow):
            x = cx0 + 3 + i * (bar_w + gap)
            bh = int(v / y_flow_max * ch) if y_flow_max > 0 else 0
            if bh > 0:
                cv2.rectangle(panel, (x, cy1 - bh), (x + bar_w, cy1), GREEN, -1)

    N = len(flow)
    if N > 1:
        n_ticks = min(6, N)
        for k in range(n_ticks):
            idx = int(k * (N - 1) / max(n_ticks - 1, 1))
            x = cx0 + int(idx * cw / max(N - 1, 1))
            cv2.line(panel, (x, cy1), (x, cy1 + 4), TEXT_3, 1)
            cv2.putText(panel, str(idx), (x - 10, cy1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, TEXT_3, 1)
        cv2.putText(panel, 'Frame', (cx0 + cw // 2 - 18, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_3, 1)

    # Legend
    lx, ly = cx0 + 8, cy0 + 4
    cv2.rectangle(panel, (lx, ly), (lx + 10, ly + 10), GREEN, -1)
    cv2.putText(panel, 'Vehicles/frame', (lx + 15, ly + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_2, 1)

    cv2.rectangle(panel, (0, 0), (W - 1, H - 1), BORDER, 1)
    return panel


# ═══════════════════════════════════════════════════════════════
# Dashboard Renderer
# ═══════════════════════════════════════════════════════════════
class DashboardRenderer:
    def __init__(self, W=1200, H=600):
        self.W, self.H = W, H
        raw_det_w = W - W * 2 // 5
        self.det_w = raw_det_w // 2 * 2    # force even width for side-by-side subpanels
        self.panel_w = W - self.det_w      # adjust panel width to absorb rounding pixel
        self.det_h = H * 3 // 5            # detection row height
        self.bev_h = H - self.det_h        # BEV height
        self.half_w = self.det_w // 2      # half (always det_w // 2)

    def render(self, img, tracks, analysis, timing, quality, calib_K=None, heatmap=None):
        # Top-left row: 3D detection | Heatmap (side by side)
        det = draw_det_panel(img, tracks, self.half_w, self.det_h, calib_K)
        hm = draw_heatmap_panel(heatmap, self.half_w, self.det_h)
        top_row = np.hstack([det, hm])

        # Bottom-left: BEV
        bev = BEVRenderer(self.det_w, self.bev_h).render(tracks)
        left = np.vstack([top_row, bev])

        # Right column: stats + flow trend
        td = analysis.get('trend', {})
        stats_h = (self.H - self.bev_h) // 2
        trend_h = self.H - stats_h

        s = draw_stats(self.panel_w, stats_h, analysis, timing, quality)
        t = draw_trend(self.panel_w, trend_h, td.get('flow_counts', []))
        right = np.vstack([s, t])

        dashboard = np.hstack([left, right])

        title = 'RTM3D V5 - Roadside Monocular 3D Detection'
        cv2.putText(dashboard, title, (self.det_w + 16, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_2, 1)

        return dashboard
