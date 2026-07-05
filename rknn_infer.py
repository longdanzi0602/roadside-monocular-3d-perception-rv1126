#!/usr/bin/env python3
"""
RV1126B board-side inference for RTM3D V5.
Uses RKNNLite runtime + cpu_postprocess for 3D decoding.
"""
import os, sys, time, json
import numpy as np
import cv2

# ---- RKNN Lite ----
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    print("WARN: rknnlite not available, using mock mode")
    HAS_RKNN = False
    class RKNNLite:
        def __init__(self): pass
        def load_rknn(self, path): return 0
        def init_runtime(self, core_mask=0): return 0
        def inference(self, inputs): return [np.zeros((1, c, 128, 424), dtype=np.float32) for c in [3,2,18,8,3,1,2,9,2]]
        def release(self): pass

# ---- CPU postprocess ----
from cpu_postprocess import car_pose_decode_np

INPUT_H, INPUT_W = 512, 1696
STRIDE = 4


class RTM3DInference:
    """V5 RTM3D inference on RV1126B NPU."""

    def __init__(self, rknn_path, calib_json=None):
        self.rknn = RKNNLite()
        print(f"Loading RKNN: {rknn_path}")
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO if hasattr(RKNNLite, 'NPU_CORE_AUTO') else 0)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        print("NPU ready.")

        # Load calibration (per-scene or single)
        self.calib = None
        if calib_json and os.path.exists(calib_json):
            with open(calib_json) as f:
                calib_data = json.load(f)
            K = np.array(calib_data['cam_K'], dtype=np.float32).reshape(3, 3)
            self.calib = K
            print(f"Calib loaded: fx={K[0,0]:.1f}")

    def set_calib(self, K):
        """Set intrinsic matrix for current scene."""
        self.calib = np.array(K, dtype=np.float32).reshape(3, 3)

    def preprocess(self, img_bgr):
        """Letterbox + return BGR uint8 (NPU normalizes internally)."""
        h, w = img_bgr.shape[:2]
        scale = min(INPUT_H / h, INPUT_W / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (nw, nh))

        canvas = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        dy = (INPUT_H - nh) // 2
        dx = (INPUT_W - nw) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized

        return canvas, scale, dx, dy

    def infer(self, img_bgr, conf_thresh=0.15, K=200):
        """Full pipeline: preprocess → NPU → decode → detections."""
        t0 = time.time()

        # Preprocess
        inp, scale, dx, dy = self.preprocess(img_bgr)
        inp_npu = np.expand_dims(inp, axis=0)  # [1, 512, 1696, 3] BGR uint8
        t_prep = time.time()

        # NPU inference
        npu_outputs = self.rknn.inference(inputs=[inp_npu])
        t_npu = time.time()

        # Parse NPU outputs into dict
        hm_raw = npu_outputs[0][0]       # [3, 128, 424]
        # Apply sigmoid to convert raw NPU logits to [0, 1] probabilities
        hm_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(hm_raw, -15, 15)))
        np_outputs = {
            'hm': hm_sigmoid,
            'wh': npu_outputs[1][0],      # [2, 128, 424]
            'hps': npu_outputs[2][0],     # [18, 128, 424]
            'rot': npu_outputs[3][0],     # [8, 128, 424]
            'dim': npu_outputs[4][0],     # [3, 128, 424]
            'prob': npu_outputs[5][0],    # [1, 128, 424]
            'reg': npu_outputs[6][0],     # [2, 128, 424]
            'hm_hp': npu_outputs[7][0],   # [9, 128, 424]
            'hp_offset': npu_outputs[8][0], # [2, 128, 424]
        }

        # Calibration (model-space)
        if self.calib is not None:
            K_model = self.calib.copy() * scale
            K_model[0, 2] += dx
            K_model[1, 2] += dy
            calib_arr = np.zeros((3, 4), dtype=np.float32)
            calib_arr[:3, :3] = K_model
            np_outputs['calib'] = calib_arr

        # Decode
        detections = car_pose_decode_np(np_outputs, K=K, conf_thresh=conf_thresh)
        t_post = time.time()

        # Map bboxes back to original image coords
        for det in detections:
            bb = det['bbox_2d']
            bb['xmin'] = (bb['xmin'] - dx) / scale
            bb['ymin'] = (bb['ymin'] - dy) / scale
            bb['xmax'] = (bb['xmax'] - dx) / scale
            bb['ymax'] = (bb['ymax'] - dy) / scale

        timing = {
            'prep_ms': (t_prep - t0) * 1000,
            'npu_ms': (t_npu - t_prep) * 1000,
            'post_ms': (t_post - t_npu) * 1000,
            'total_ms': (t_post - t0) * 1000,
            'fps': 1.0 / max(t_post - t0, 1e-6),
        }

        return detections, timing, hm_sigmoid

    def release(self):
        self.rknn.release()


if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='rtm3d_v5_512x1696_int8.rknn')
    parser.add_argument('--image', help='Test image path')
    parser.add_argument('--calib', help='Calibration JSON for this scene')
    parser.add_argument('--conf', type=float, default=0.15)
    args = parser.parse_args()

    infer = RTM3DInference(args.model, calib_json=args.calib)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Cannot read: {args.image}")
        else:
            dets, timing = infer.infer(img, conf_thresh=args.conf)
            print(f"\n{len(dets)} detections")
            print(f"Timing: prep={timing['prep_ms']:.0f}ms npu={timing['npu_ms']:.0f}ms post={timing['post_ms']:.0f}ms total={timing['total_ms']:.0f}ms ({timing['fps']:.1f} fps)")
            for d in dets:
                loc = d['location_3d']
                print(f"  {d['class']} conf={d['confidence']:.2f} pos=({loc['x']:.1f},{loc['y']:.1f},{loc['z']:.1f})m")
    else:
        # Quick test with blank frame
        print("No image specified, running quick test with blank frame...")
        blank = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        dets, timing = infer.infer(blank, conf_thresh=0.15)
        print(f"Test: {len(dets)} dets, {timing['total_ms']:.0f}ms")

    infer.release()
