#!/usr/bin/env python3
"""
主入口: 端侧 AI 视觉应用完整流水线.
考核②: 视觉算法性能 (帧率/准确率/速度)
考核④: 长时间高负载运行下的系统稳定性
"""

import os, sys, json, time, argparse, traceback
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict, deque

BOARD_MODE = False
try:
    from rknnlite.api import RKNNLite
    BOARD_MODE = True
except ImportError:
    BOARD_MODE = False

DARK_BG = (18, 22, 30)
COLORS = {'Car': (0, 255, 0), 'Pedestrian': (0, 165, 255), 'Cyclist': (255, 255, 0)}


# ================================================================
# Simple IoU Tracker
# ================================================================
def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x1 >= x2 or y1 >= y2: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


class SimpleTracker:
    def __init__(self, max_age=3, iou_thresh=0.2):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.next_id = 1
        self.tracks = {}  # id -> {cls, bbox, loc_3d, age, velocity}

    def update(self, detections):
        # Build bbox list from detections
        det_boxes = []
        for d in detections:
            bb = d.get('bbox_2d', {})
            det_boxes.append([bb.get('xmin', 0), bb.get('ymin', 0),
                              bb.get('xmax', 0), bb.get('ymax', 0)])

        # Match
        active_ids = list(self.tracks.keys())
        matched_tids = set()
        matched_dids = set()

        for ti, tid in enumerate(active_ids):
            t = self.tracks[tid]
            best_i, best_d = 0, -1
            for di, db in enumerate(det_boxes):
                if di in matched_dids: continue
                tb = [t['bbox'][0], t['bbox'][1], t['bbox'][2], t['bbox'][3]]
                i = _iou(tb, db)
                if i > best_i:
                    best_i = i
                    best_d = di
            if best_i > self.iou_thresh:
                matched_tids.add(tid)
                matched_dids.add(best_d)
                # Update track
                det = detections[best_d]
                prev_loc = t['loc_3d']
                prev_cy = (t['bbox'][1] + t['bbox'][3]) / 2.0  # save before update
                new_loc = det.get('location_3d', {})
                t['bbox'] = det_boxes[best_d]
                new_cy = (det_boxes[best_d][1] + det_boxes[best_d][3]) / 2.0
                t['prev_center_y'] = prev_cy
                t['center_y'] = new_cy
                t['loc_3d'] = new_loc
                t['cls'] = det.get('class', 'Car')
                t['age'] = 0
                t['confidence'] = det.get('confidence', 0)
                t['dimensions_3d'] = det.get('dimensions_3d', {})
                t['yaw'] = det.get('yaw', 0)
                t['bbox_2d'] = det.get('bbox_2d', {})
                # Velocity
                dx = new_loc.get('x', 0) - prev_loc.get('x', 0)
                dy = new_loc.get('y', 0) - prev_loc.get('y', 0)
                dz = new_loc.get('z', 0) - prev_loc.get('z', 0)
                t['velocity'] = {
                    'vx': dx, 'vz': dz,
                    'speed_kmh': np.sqrt(dx**2 + dz**2) * 5.0 * 3.6
                }

        # Age unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_tids:
                self.tracks[tid]['age'] += 1
                self.tracks[tid]['velocity'] = {'vx': 0, 'vz': 0, 'speed_kmh': 0}
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        # Create new tracks
        for di, d in enumerate(detections):
            if di not in matched_dids:
                bb = det_boxes[di]
                cy = (bb[1] + bb[3]) / 2.0
                self.tracks[self.next_id] = {
                    'bbox': bb,
                    'loc_3d': d.get('location_3d', {}),
                    'cls': d.get('class', 'Car'),
                    'age': 0,
                    'confidence': d.get('confidence', 0),
                    'dimensions_3d': d.get('dimensions_3d', {}),
                    'yaw': d.get('yaw', 0),
                    'bbox_2d': d.get('bbox_2d', {}),
                    'velocity': {'vx': 0, 'vz': 0, 'speed_kmh': 0},
                    'center_y': cy,
                    'prev_center_y': cy,  # no history yet, set = current
                }
                self.next_id += 1

        # Return confirmed tracks as list
        result = []
        for tid, t in self.tracks.items():
            result.append({
                'track_id': tid,
                'class': t['cls'],
                'bbox_2d': t['bbox_2d'],
                'location_3d': t['loc_3d'],
                'dimensions_3d': t['dimensions_3d'],
                'yaw': t['yaw'],
                'confidence': t['confidence'],
                'velocity': t['velocity'],
                'center_y': t.get('center_y', 0),
                'prev_center_y': t.get('prev_center_y', 0),
            })
        return result


# ================================================================
# Performance Recorder
# ================================================================
class PerformanceRecorder:
    def __init__(self, log_interval=100):
        self.start_time = time.time()
        self.frame_times = []
        self.prep_times = []
        self.npu_times = []
        self.post_times = []
        self.anomalies = []
        self.errors = []
        self.log_interval = log_interval
        self.frame_count = 0

    def record(self, timing_dict):
        elapsed_ms = timing_dict['total_ms']
        self.frame_times.append(elapsed_ms)
        self.prep_times.append(timing_dict.get('prep_ms', 0))
        self.npu_times.append(timing_dict.get('npu_ms', 0))
        self.post_times.append(timing_dict.get('post_ms', 0))
        self.frame_count += 1
        if len(self.frame_times) > 50:
            med = np.median(self.frame_times)
            if elapsed_ms > med * 3:
                self.anomalies.append((self.frame_count, 'latency_spike',
                                       f'{elapsed_ms:.0f}ms vs median {med:.0f}ms'))

    def record_error(self, msg):
        self.errors.append((self.frame_count, msg))

    def report(self):
        elapsed = time.time() - self.start_time
        times = self.frame_times
        if not times:
            return {'total_frames': 0, 'avg_fps': 0, 'latency_ms': {},
                    'latency_breakdown_avg_ms': {},
                    'anomalies': 0, 'errors': 0, 'stability_score': 0}
        return {
            'total_frames': self.frame_count,
            'elapsed_seconds': round(elapsed, 1),
            'avg_fps': round(len(times) / elapsed, 2) if elapsed > 0 else 0,
            'latency_ms': {
                'mean': round(np.mean(times), 1),
                'median': round(np.median(times), 1),
                'p99': round(np.percentile(times, 99), 1),
                'min': round(min(times), 1),
                'max': round(max(times), 1),
            },
            'latency_breakdown_avg_ms': {
                'preprocess': round(np.mean(self.prep_times), 1) if self.prep_times else 0,
                'npu': round(np.mean(self.npu_times), 1) if self.npu_times else 0,
                'postprocess': round(np.mean(self.post_times), 1) if self.post_times else 0,
            },
            'anomalies': len(self.anomalies),
            'errors': len(self.errors),
            'stability_score': round(max(0, 100 - len(self.anomalies) * 5 -
                                         len(self.errors) * 10), 1),
        }


# ================================================================
# Main Pipeline
# ================================================================
class MainPipeline:
    def __init__(self, args):
        self.args = args
        self.recorder = PerformanceRecorder(log_interval=args.log_interval)
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_server(self):
        sys.path.insert(0, os.path.dirname(__file__))
        from model_rtm3d import create_rtm3d_resnet18
        from cpu_postprocess import car_pose_decode_np
        import torch
        self.car_pose_decode_np = car_pose_decode_np
        heads = {'hm': 3, 'wh': 2, 'hps': 18, 'rot': 8, 'dim': 3,
                 'prob': 1, 'reg': 2, 'hm_hp': 9, 'hp_offset': 2}
        model = create_rtm3d_resnet18(heads, head_conv=128, pretrained=False)
        ckpt = torch.load(self.args.model, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        self.model = model.cuda().eval()
        self.torch = torch

    def _init_board(self):
        sys.path.insert(0, os.path.dirname(__file__))
        from rknn_infer import RTM3DInference
        self.detector = RTM3DInference(self.args.model)

    def process_frame_server(self, img_bgr, frame_id=None):
        from cpu_postprocess import car_pose_decode_np
        import torch
        t0 = time.time()
        INPUT_H, INPUT_W = 512, 1696
        h, w = img_bgr.shape[:2]
        scale = min(INPUT_H / h, INPUT_W / w)
        nh, nw = int(h * scale), int(w * scale)

        # Optimized: resize uint8 first (6MB→0.8MB), then convert to float32
        rs = cv2.resize(img_bgr, (nw, nh))  # uint8, fast
        dy, dx = (INPUT_H - nh) // 2, (INPUT_W - nw) // 2

        # Directly build float32 canvas with padding, normalize in one pass
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        canvas = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.float32)
        # Convert only the valid region: uint8→float, /255, normalize
        roi = rs.astype(np.float32) * (1.0 / 255.0)
        roi = (roi - mean) / std
        canvas[dy:dy + nh, dx:dx + nw] = roi

        inp = torch.from_numpy(canvas.transpose(2, 0, 1)).unsqueeze(0).cuda()
        t_prep = time.time()

        with torch.no_grad():
            out = self.model(inp)
            out = out[-1] if isinstance(out, list) else out
        t_npu = time.time()

        # Calibration
        K_model = None
        K_raw = None
        if self.args.calib_dir:
            fid = frame_id if frame_id else '000032'
            calib_path = os.path.join(self.args.calib_dir, f'camera_intrinsic_{fid}.json')
            if os.path.exists(calib_path):
                with open(calib_path) as f:
                    K_mat = np.array(json.load(f)['cam_K'], dtype=np.float32).reshape(3, 3)
                K_raw = K_mat.copy()
                K_model = K_mat.copy() * scale
                K_model[0, 2] += dx
                K_model[1, 2] += dy

        hm_np = torch.sigmoid(out['hm'][0]).cpu().numpy()
        np_out = {
            'hm': hm_np,
            'wh': out['wh'][0].cpu().numpy(),
            'hps': out['hps'][0].cpu().numpy(),
            'dim': out['dim'][0].cpu().numpy(),
            'rot': out['rot'][0].cpu().numpy(),
            'prob': out['prob'][0].cpu().numpy(),
            'reg': out['reg'][0].cpu().numpy(),
            'hm_hp': out['hm_hp'][0].cpu().numpy(),
            'hp_offset': out['hp_offset'][0].cpu().numpy(),
        }
        if K_model is not None:
            cal_arr = np.zeros((3, 4), dtype=np.float32)
            cal_arr[:3, :3] = K_model
            np_out['calib'] = cal_arr

        dets = car_pose_decode_np(np_out, K=self.args.topk, conf_thresh=self.args.conf)
        for det in dets:
            bb = det['bbox_2d']
            bb['xmin'] = (bb['xmin'] - dx) / scale
            bb['ymin'] = (bb['ymin'] - dy) / scale
            bb['xmax'] = (bb['xmax'] - dx) / scale
            bb['ymax'] = (bb['ymax'] - dy) / scale

        t_post = time.time()
        timing = {
            'prep_ms': (t_prep - t0) * 1000,
            'npu_ms': (t_npu - t_prep) * 1000,
            'post_ms': (t_post - t_npu) * 1000,
            'total_ms': (t_post - t0) * 1000,
            'fps': 1.0 / max(t_post - t0, 1e-6),
        }
        return dets, timing, K_raw, hm_np

    def run_image_dir(self):
        from image_enhance import ImageEnhancer
        from visualization import DashboardRenderer
        from traffic_analyzer import TrafficCounter, DensityGrid

        img_dir = Path(self.args.image_dir)
        image_files = sorted(img_dir.glob('*.jpg'))
        if not image_files:
            print(f"No images found in {img_dir}")
            return
        n_total = len(image_files)
        print(f"Processing {n_total} images...")

        enhancer = ImageEnhancer()
        counter = TrafficCounter(line_y=540)
        density = DensityGrid()
        tracker = SimpleTracker(max_age=3, iou_thresh=0.2)
        if BOARD_MODE or self.args.board:
            # Pixel-perfect rendering for 7-inch LCD screen (1024x600)
            viz = DashboardRenderer(W=1024, H=600)
        else:
            viz = DashboardRenderer(W=1200, H=600)

        fb_dev = None
        if getattr(self.args, 'fb', False):
            try:
                fb_dev = open('/dev/fb0', 'wb')
                print("Direct Framebuffer writing enabled on /dev/fb0.")
            except Exception as e:
                print(f"ERROR: Failed to open /dev/fb0: {e}. Framebuffer output disabled.")

        # History for trends
        flow_history = deque(maxlen=120)
        class_total = defaultdict(int)
        seen_track_ids = set()
        all_detections = []

        for i, img_path in enumerate(image_files):
            fid = img_path.stem
            img = cv2.imread(str(img_path))
            if img is None:
                self.recorder.record_error(f'failed to read {img_path}')
                continue
            
            img_raw = img.copy()  # Save a copy of the raw, un-enhanced image

            try:
                # 1. Enhancement
                if self.args.enhance:
                    img, q_before, q_after = enhancer.enhance(img)
                else:
                    q_before = enhancer._measure_quality(img)
                    q_after = q_before

                # 2. Detection
                t0 = time.time()
                if BOARD_MODE:
                    # Load calibration for this frame (if available)
                    K_raw = None
                    if self.args.calib_dir:
                        calib_path = os.path.join(self.args.calib_dir, f'camera_intrinsic_{fid}.json')
                        # Fallback if standard camera_intrinsic_ prefix is missing
                        if not os.path.exists(calib_path):
                            calib_path = os.path.join(self.args.calib_dir, f'{fid}.json')
                            
                        if os.path.exists(calib_path):
                            with open(calib_path) as cf:
                                K_mat = np.array(json.load(cf)['cam_K'], dtype=np.float32).reshape(3, 3)
                            K_raw = K_mat.copy()
                            self.detector.set_calib(K_mat)
                    dets, timing, hm_sigmoid = self.detector.infer(img, conf_thresh=self.args.conf, K=self.args.topk)
                    hm_np = hm_sigmoid
                else:
                    dets, timing, K_raw, hm_np = self.process_frame_server(img, frame_id=fid)

                # 3. Tracking (for traffic analysis only — not visualization)
                tracks = tracker.update(dets)

                # Convert raw dets to track-like format for visualization (no stale tracks)
                raw_tracks = []
                for di, det in enumerate(dets):
                    raw_tracks.append({
                        'track_id': di,
                        'class': det.get('class', 'Car'),
                        'bbox_2d': det.get('bbox_2d', {}),
                        'location_3d': det.get('location_3d', {}),
                        'dimensions_3d': det.get('dimensions_3d', {}),
                        'yaw': det.get('yaw', 0),
                        'confidence': det.get('confidence', 0),
                        'velocity': {'vx': 0, 'vz': 0, 'speed_kmh': 0},
                    })

                # 4. Traffic analysis
                counter.update(tracks)
                density.update(tracks)
                for t in tracks:
                    tid = t.get('track_id')
                    cls = t.get('class', 'Car')
                    if tid not in seen_track_ids:
                        seen_track_ids.add(tid)
                        class_total[cls] += 1

                car_count = sum(1 for t in tracks if t.get('class') == 'Car')
                flow_history.append(car_count)

                analysis = {
                    'flow_count': {
                        'total': counter.flow_rate(),
                        'count_in': counter.counts.get('Car', 0),
                        'count_out': 0,
                        'class_in': dict(counter.counts),
                    },
                    'class_total': dict(class_total),
                    'active_targets': len(tracks),
                    'occupancy_rate': len(tracks) / max(
                        (abs(density.world_range[0][1] - density.world_range[0][0]) *
                         abs(density.world_range[1][1] - density.world_range[1][0])),
                        1.0),
                    'max_density': int(density.grid.max()),
                    'density_grid': density.grid.copy().tolist(),
                    'trend': {
                        'flow_counts': list(flow_history),
                    },
                    'runtime_sec': time.time() - self.recorder.start_time,
                    'total_frames': i + 1,
                    'trend_points': len(flow_history),
                }

                # 5. Quality dict
                quality = {
                    'brightness': q_after.brightness,
                    'contrast': q_after.contrast,
                    'is_dark': q_after.brightness < 80,
                    'is_overexp': q_after.brightness > 220,
                }

                # Visualization with raw detections (no stale tracker boxes)
                if self.args.show or getattr(self.args, 'save_img', False) or fb_dev is not None:
                    dash = viz.render(img, raw_tracks, analysis, timing, quality, calib_K=K_raw, heatmap=hm_np)

                    # Write to framebuffer
                    if fb_dev is not None:
                        try:
                            # Convert to BGRA format for standard 32-bit framebuffer
                            dash_bgra = cv2.cvtColor(dash, cv2.COLOR_BGR2BGRA)
                            fb_dev.seek(0)
                            fb_dev.write(dash_bgra.tobytes())
                            fb_dev.flush()
                        except Exception as fe:
                            print(f"WARN: Failed to write to framebuffer: {fe}")

                    # 7. Save
                    if getattr(self.args, 'save_img', False):
                        out_path = self.output_dir / f'{fid}_dashboard.jpg'
                        cv2.imwrite(str(out_path), dash)
                        
                        # Save separate 3D detection and heatmap panels for reports
                        try:
                            from visualization import draw_det_panel, draw_heatmap_panel
                            det_panel = draw_det_panel(img, raw_tracks, viz.half_w, viz.det_h, calib_K=K_raw)
                            hm_panel = draw_heatmap_panel(hm_np, viz.half_w, viz.det_h)
                            
                            # Original raw image resized to match the panels
                            orig_raw_resized = cv2.resize(img_raw, (viz.half_w, viz.det_h))
                            # Enhanced image resized to match the panels
                            orig_enhanced_resized = cv2.resize(img, (viz.half_w, viz.det_h))
                            
                            # Save separate panels
                            cv2.imwrite(str(self.output_dir / f'{fid}_det.jpg'), det_panel)
                            cv2.imwrite(str(self.output_dir / f'{fid}_heatmap.jpg'), hm_panel)
                            
                            # Create and save a beautiful triple-panel image (Original Enhanced | 3D Detection | Heatmap)
                            triple_panel = np.hstack([orig_enhanced_resized, det_panel, hm_panel])
                            cv2.imwrite(str(self.output_dir / f'{fid}_triple.jpg'), triple_panel)
                            
                            # Create and save a beautiful quad-panel image (Original Raw | Enhanced | 3D Detection | Heatmap)
                            quad_panel = np.hstack([orig_raw_resized, orig_enhanced_resized, det_panel, hm_panel])
                            cv2.imwrite(str(self.output_dir / f'{fid}_quad.jpg'), quad_panel)
                            
                            print(f"  Saved: {fid}_det.jpg, {fid}_heatmap.jpg, {fid}_triple.jpg, and {fid}_quad.jpg to {self.output_dir}")
                        except Exception as se:
                            print(f"  WARN: Failed to save separate panels: {se}")

                # Real-time display on LCD screen
                if self.args.show:
                    try:
                        cv2.namedWindow("RTM3D V5 Dashboard", cv2.WINDOW_NORMAL)
                        if BOARD_MODE or self.args.board:
                            # Fullscreen to eliminate window borders/title bar and achieve pixel-perfect native display
                            cv2.setWindowProperty("RTM3D V5 Dashboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.resizeWindow("RTM3D V5 Dashboard", 1200, 600)
                        cv2.imshow("RTM3D V5 Dashboard", dash)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            print("User cancelled visualization display.")
                    except Exception as ve:
                        print(f"WARN: Failed to show GUI window: {ve}")

                self.recorder.record(timing)
                all_detections.extend(dets)

                if i < 3 or (i + 1) % 50 == 0:
                    print(f"  [{i+1:4d}/{n_total}] {fid}: {len(raw_tracks)} dets "
                          f"({len(tracks)} tracks) {timing['fps']:.1f}fps")

            except Exception as e:
                self.recorder.record_error(f'{fid}: {e}')
                traceback.print_exc()

        return self._generate_report(n_total, all_detections)

    def _generate_report(self, n_frames, all_detections):
        perf = self.recorder.report()
        cars = [d for d in all_detections if d.get('class') == 'Car']
        confs = [d.get('confidence', 0) for d in cars]
        depths = []
        for d in cars:
            loc = d.get('location_3d', {})
            d3 = np.sqrt(loc.get('x', 0)**2 + loc.get('y', 0)**2 + loc.get('z', 0)**2)
            if d3 > 0:
                depths.append(d3)

        report = {
            'version': 'V5 RTM3D',
            'board': 'RV1126B (simulated)' if not BOARD_MODE else 'RV1126B NPU',
            'total_frames': n_frames,
            'total_detections': len(all_detections),
            'total_cars': len(cars),
            'avg_det_per_frame': round(len(all_detections) / max(n_frames, 1), 1),
            'confidence': {
                'mean': round(np.mean(confs), 3) if confs else 0,
                'median': round(np.median(confs), 3) if confs else 0,
            },
            'depth_range_m': {
                'median': round(np.median(depths), 1) if depths else 0,
                'mean': round(np.mean(depths), 1) if depths else 0,
            },
            'performance': perf,
            'anomaly_details': self.recorder.anomalies[:10],
            'error_details': self.recorder.errors,
        }
        report_path = self.output_dir / 'report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save raw frame-by-frame latency breakdown to CSV
        latencies_path = self.output_dir / 'latencies.csv'
        with open(latencies_path, 'w') as f:
            f.write("frame_idx,prep_ms,npu_ms,post_ms,total_ms\n")
            for idx in range(len(self.recorder.frame_times)):
                f.write(f"{idx+1},"
                        f"{self.recorder.prep_times[idx]:.2f},"
                        f"{self.recorder.npu_times[idx]:.2f},"
                        f"{self.recorder.post_times[idx]:.2f},"
                        f"{self.recorder.frame_times[idx]:.2f}\n")

        times_list = self.recorder.frame_times
        p50 = round(np.percentile(times_list, 50), 1) if times_list else 0
        p90 = round(np.percentile(times_list, 90), 1) if times_list else 0
        p95 = round(np.percentile(times_list, 95), 1) if times_list else 0
        p99 = round(np.percentile(times_list, 99), 1) if times_list else 0

        print(f"\n{'='*60}")
        print(f"Competition Report")
        print(f"{'='*60}")
        print(f"Frames:        {n_frames}")
        print(f"Detections:    {len(all_detections)} "
              f"({report['avg_det_per_frame']}/frame)")
        print(f"Avg FPS:       {perf['avg_fps']}")
        print(f"Latency Percentiles:")
        print(f"  P50 (Median): {p50}ms")
        print(f"  P90:          {p90}ms")
        print(f"  P95:          {p95}ms")
        print(f"  P99:          {p99}ms")
        print(f"Avg Stage Breakdown:")
        print(f"  Preprocess:   {perf['latency_breakdown_avg_ms']['preprocess']}ms")
        print(f"  NPU Inference:{perf['latency_breakdown_avg_ms']['npu']}ms")
        print(f"  Postprocess:  {perf['latency_breakdown_avg_ms']['postprocess']}ms")
        print(f"Stability:     {perf['stability_score']}/100")
        print(f"Errors:        {perf['errors']}")
        print(f"Report saved:  {report_path}")
        print(f"Raw CSV saved: {latencies_path}")
        print(f"{'='*60}")
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RTM3D V5 Competition Pipeline')
    parser.add_argument('--image-dir', help='Input image directory')
    parser.add_argument('--video', help='Input video file')
    parser.add_argument('--calib-dir', help='Calibration directory')
    parser.add_argument('--model', default='N-RTM3D-int8.rknn')
    parser.add_argument('--output', default='output/', help='Output directory')
    parser.add_argument('--board', action='store_true')
    parser.add_argument('--enhance', action='store_true')
    parser.add_argument('--show', action='store_true', help='Show visualization on screen')
    parser.add_argument('--fb', action='store_true', help='Direct write visualization to Linux framebuffer /dev/fb0')
    parser.add_argument('--save-img', action='store_true', help='Save visualization images to disk')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--topk', type=int, default=200, help='K candidates for heatmap peak extraction (lower=faster, higher=better recall in dense scenes)')
    args = parser.parse_args()

    pipeline = MainPipeline(args)
    if args.board or BOARD_MODE:
        print("Initializing board (RKNN) runtime...")
        pipeline._init_board()
    else:
        print("Initializing server (PyTorch) runtime...")
        pipeline._init_server()

    if args.image_dir:
        pipeline.run_image_dir()
    elif args.video:
        print("Video mode not yet implemented, use --image-dir")
    else:
        print("Usage: python main_pipeline.py --image-dir <dir> [--enhance]")
