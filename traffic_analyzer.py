#!/usr/bin/env python3
"""考核核心: 过线计数 / 速度统计 / 密度网格 / 趋势分析."""
import numpy as np
from collections import defaultdict, deque
import time

class TrafficCounter:
    """Virtual loop line-crossing counter."""
    def __init__(self, line_y=540, direction='down'):
        self.line_y = line_y
        self.direction = direction  # 'down' means crossing from above to below
        self.counts = defaultdict(int)  # per class
        self.crossed_ids = set()
        self.history = deque(maxlen=600)  # per-second counts

    def update(self, tracked_objects, timestamp=None):
        """Update crossing counts from tracked objects."""
        for obj in tracked_objects:
            tid = obj.get('track_id', id(obj))
            if tid in self.crossed_ids:
                continue
            cy = obj.get('center_y', 0)
            prev_cy = obj.get('prev_center_y', cy)
            cls = obj.get('class', 'unknown')
            if self.direction == 'down':
                crossed = prev_cy < self.line_y <= cy
            else:
                crossed = prev_cy > self.line_y >= cy
            if crossed:
                self.counts[cls] += 1
                self.crossed_ids.add(tid)
                if timestamp is None:
                    timestamp = time.time()
                self.history.append((timestamp, cls))
        return dict(self.counts)

    def flow_rate(self, window_seconds=60):
        """Vehicles per minute in recent window."""
        now = time.time()
        cutoff = now - window_seconds
        recent = [t for t, _ in self.history if t > cutoff]
        return len(recent) / max(window_seconds / 60, 0.1)


class SpeedEstimator:
    """Kalman-filtered speed from 3D positions."""
    def __init__(self, fps=5.0, alpha=0.3):
        self.fps = fps
        self.alpha = alpha  # EMA smoothing
        self.speeds = defaultdict(lambda: {'v': 0, 'history': deque(maxlen=10)})

    def update(self, obj_id, location_3d):
        """Estimate speed in km/h from consecutive positions."""
        x, y, z = location_3d['x'], location_3d['y'], location_3d['z']
        prev = self.speeds[obj_id]
        prev_loc = prev.get('last_loc')
        if prev_loc:
            dist = np.sqrt(
                (x - prev_loc[0])**2 + (y - prev_loc[1])**2 + (z - prev_loc[2])**2
            )
            v_ms = dist * self.fps
            v_kph = v_ms * 3.6
            prev['v'] = prev['v'] * (1 - self.alpha) + v_kph * self.alpha
            prev['history'].append(prev['v'])
        prev['last_loc'] = (x, y, z)
        return prev['v']


class DensityGrid:
    """Spatial traffic density heatmap."""
    def __init__(self, grid_size=(10, 6), world_range=((-100, 100), (0, 200))):
        self.grid_size = grid_size
        self.world_range = world_range
        self.grid = np.zeros(grid_size, dtype=np.float32)
        self.decay = 0.95  # per-frame decay

    def update(self, detections):
        """Accumulate vehicle positions into density grid."""
        self.grid *= self.decay
        x_range = self.world_range[0]
        z_range = self.world_range[1]
        dx = (x_range[1] - x_range[0]) / self.grid_size[1]
        dz = (z_range[1] - z_range[0]) / self.grid_size[0]
        for det in detections:
            loc = det.get('location_3d', {})
            x, z = loc.get('x', 0), loc.get('z', 0)
            gx = int((x - x_range[0]) / dx)
            gz = int((z - z_range[0]) / dz)
            if 0 <= gx < self.grid_size[1] and 0 <= gz < self.grid_size[0]:
                self.grid[gz, gx] += 1.0
        return self.grid.copy()


class TrendAnalyzer:
    """Traffic flow trend analysis over time windows."""
    def __init__(self, window_minutes=5):
        self.window = window_minutes
        self.buckets = deque(maxlen=window_minutes * 2)  # 30s buckets

    def add_sample(self, n_vehicles, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        self.buckets.append((timestamp, n_vehicles))

    def summary(self):
        if len(self.buckets) < 2:
            return {'trend': 'stable', 'current': 0, 'avg': 0, 'peak': 0}
        times = [t for t, _ in self.buckets]
        counts = [c for _, c in self.buckets]
        now = time.time()
        recent = [c for t, c in self.buckets if now - t < 120]
        current = np.mean(recent) if recent else counts[-1]
        avg = np.mean(counts)
        peak = max(counts)
        if current > avg * 1.2:
            trend = 'rising'
        elif current < avg * 0.8:
            trend = 'falling'
        else:
            trend = 'stable'
        return {'trend': trend, 'current': round(current, 1),
                'avg': round(avg, 1), 'peak': round(peak, 1)}
