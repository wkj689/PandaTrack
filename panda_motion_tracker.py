from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


logger = logging.getLogger(__name__)


class PandaMotionTracker:
    def __init__(
        self,
        reference_height: float = 0.6,
        reference_width: float = 1.2,
        fps: float = 30.0,
        shift_thr: float = 0.02,
        size_thr: float = 0.08,
    ) -> None:
        self.ref_h = reference_height
        self.ref_w = reference_width
        self.fps = fps

        # "BBox shift" thresholds
        self.shift_thr = shift_thr
        self.size_thr = size_thr

        # Tracking metrics
        self.prev_c: Optional[np.ndarray] = None
        self.pixel_per_m: Optional[float] = None

        self.total_d = 0.0
        self.avg_v = 0.0
        self.max_v = 0.0
        self.t_sum = 0.0
        self.last_ts: Optional[float] = None
        self.current_speed: float = 0.0

        # Trajectory
        self.history: List[Dict] = []

        # Tuning
        self.min_move, self.max_move = 0.01, 0.8
        self.dist_corr, self.speed_corr, self.speed_cap = 0.7, 0.7, 2.5

    # ---------------------------------------------------------
    #             Determine movement using only bbox information
    # ---------------------------------------------------------
    def bbox_shift_moving(
        self, cur_bbox: np.ndarray | None, prev_bbox: np.ndarray | None
    ) -> bool:
        if cur_bbox is None or prev_bbox is None:
            return False

        cx1 = (prev_bbox[0] + prev_bbox[2]) * 0.5
        cy1 = (prev_bbox[1] + prev_bbox[3]) * 0.5
        cx2 = (cur_bbox[0] + cur_bbox[2]) * 0.5
        cy2 = (cur_bbox[1] + cur_bbox[3]) * 0.5
        shift_px = np.hypot(cx2 - cx1, cy2 - cy1)
        diag = np.hypot(
            cur_bbox[2] - cur_bbox[0], cur_bbox[3] - cur_bbox[1]
        )
        shift_ratio = shift_px / (diag + 1e-6)

        w1, h1 = prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1]
        w2, h2 = cur_bbox[2] - cur_bbox[0], cur_bbox[3] - cur_bbox[1]
        size_ratio = abs(w2 - w1) / (w1 + 1e-6) + abs(h2 - h1) / (h1 + 1e-6)

        return (shift_ratio > self.shift_thr) or (size_ratio > self.size_thr)

    # ---------------------------------------------------------
    def _scale_px(self, bbox: np.ndarray) -> Optional[float]:
        x1, y1, x2, y2 = bbox
        hp, wp = abs(y2 - y1), abs(x2 - x1)
        if hp == 0 or wp == 0:
            return None
        s_h = hp / self.ref_h
        s_w = wp / self.ref_w
        return (0.6 * s_h + 0.4 * s_w) * self.dist_corr

    # ---------------------------------------------------------
    def update(self, bbox: np.ndarray | None, ts: float, moving: bool) -> float:
        if bbox is None:
            self.prev_c = None
            self.current_speed = 0.0
            return 0.0

        c = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])

        if self.last_ts is None:
            self.last_ts = ts
            self.prev_c = c
            self.history.append({"t": ts, "pos": c, "moving": moving})
            self.current_speed = 0.0
            return 0.0

        dt = ts - self.last_ts
        self.t_sum += dt
        self.last_ts = ts

        # Update pixel scale
        s = self._scale_px(bbox)
        if s:
            self.pixel_per_m = s if self.pixel_per_m is None else self.pixel_per_m * 0.9 + s * 0.1

        if not (moving and self.prev_c is not None and self.pixel_per_m):
            self.prev_c = c
            self.history.append({"t": ts, "pos": c, "moving": moving})
            self.current_speed = 0.0
            return 0.0

        px = float(np.linalg.norm(c - self.prev_c))
        m = px / self.pixel_per_m
        if not (self.min_move <= m <= self.max_move):
            self.prev_c = c
            self.history.append({"t": ts, "pos": c, "moving": moving})
            self.current_speed = 0.0
            return 0.0

        self.total_d += m
        v = (m / dt) * self.speed_corr if dt > 0 else 0.0
        v = min(v, self.speed_cap)
        self.max_v = max(self.max_v, v)
        self.current_speed = v
        if self.t_sum:
            self.avg_v = (self.total_d / self.t_sum) * self.speed_corr

        self.prev_c = c
        self.history.append({"t": ts, "pos": c, "moving": moving})
        return m

    def get_current_speed(self) -> float:
        return self.current_speed

    # ---------------------------------------------------------
    def generate_motion_report(self) -> Dict[str, float]:
        return {
            "total_distance": self.total_d,
            "avg_speed": self.avg_v,
            "max_speed": self.max_v,
            "movement_count": len([h for h in self.history if h.get('moving', False)]),
        }

    # ---------------------------------------------------------
    def save_trajectory_plot(
        self,
        save_path: str | Path,
        background: np.ndarray | None = None,
        dpi: int = 150,
    ) -> None:
        """
        Save scatter trajectory plot
        - Transparent dots; color indicates dwell duration Δt at that point
        - Use a faded grayscale first frame (30% brightness) as background to match the heatmap style
        - Full-height colorbar on the right; units in minutes
        """
        if len(self.history) < 2:
            raise ValueError("Not enough trajectory data to plot")

        import matplotlib.pyplot as plt
        from matplotlib import ticker
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import cv2
        import numpy as np

        xs = np.array([h["pos"][0] for h in self.history])
        ys = np.array([h["pos"][1] for h in self.history])

        # Δt (seconds): current point -> next point
        ts = [h["t"] for h in self.history]
        dts = np.diff(ts + [ts[-1]])          # Set the last point Δt to 0
        vmax = max(dts) if np.any(dts) else 1.0

        cmap = plt.get_cmap("coolwarm_r")     # Red = short dwell; Blue = long dwell
        norm = plt.Normalize(vmin=0, vmax=vmax)

        # ------------------ Plotting ------------------
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        # ----- A. Faded grayscale background (30% brightness) -----
        if background is not None:
            bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2RGB)          # Keep 3 channels
            faded = cv2.convertScaleAbs(bg_gray, alpha=0.30, beta=0)     # Darken
            ax.imshow(faded, origin="upper")
        else:
            ax.set_facecolor("black")

        # ----- B. Scatter trajectory -----
        sc = ax.scatter(
            xs,
            ys,
            c=dts,
            cmap=cmap,
            norm=norm,
            s=50,
            alpha=0.6,
            edgecolors="none",
        )

        ax.axis("off")

        # ----- C. Full-height colorbar on the right (min) -----
        cb_ax = inset_axes(
            ax,
            width="3%", height="100%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(sc, cax=cb_ax)

        def _fmt(v, _):
            return f"{v/60:.1f}"     # seconds -> minutes
        cbar.formatter = ticker.FuncFormatter(_fmt)
        cbar.set_label("Dwell time (min)")
        cbar.update_ticks()

        # ----- D. Save -----
        fig.savefig(str(save_path), transparent=True, bbox_inches="tight")
        plt.close(fig)