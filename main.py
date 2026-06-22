#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Panda Monitor
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
import re
import subprocess
import tempfile
import os
import os.path
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

import cv2
import torch
import yaml
from ultralytics import YOLO

from panda_motion_tracker import PandaMotionTracker
from database_manager import PandaDatabaseManager  # New database manager

# --------------------------- Matplotlib font settings ---------------------------
import matplotlib

matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",  # WenQuanYi Micro Hei
    "WenQuanYi Zen Hei",  # WenQuanYi Zen Hei
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# --------------------------- Logging ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
#                        Core class
# ============================================================
class EnhancedPandaMonitor:
    def __init__(self, model_path: str, cfg: Dict) -> None:
        self.cfg = cfg
        self.det_cfg = cfg["detection"]
        self.mot_cfg = cfg["motion"]
        self.gen_cfg = cfg["general"]
        self.grid_cfg = cfg["grid_analysis"]

        logger.info("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.panda_cls_id = self._get_panda_cls_id()

        # Initialize database manager
        self.db_manager = PandaDatabaseManager(self.gen_cfg["database_path"])

        # Statistics
        self.absent_t = self.static_t = self.moving_t = 0.0
        self.last_video_time: float | None = None
        self.frame_total = 0
        self.frame_detect = 0

        # Heatmap
        self.background: np.ndarray | None = None
        self.heatmap: np.ndarray | None = None

        # Motion tracking
        self.tracker = PandaMotionTracker(
            shift_thr=self.mot_cfg["bbox_shift_threshold"],
            size_thr=self.mot_cfg["bbox_size_change"],
        )
        self._prev_bbox: np.ndarray | None = None
        self._current_bbox: np.ndarray | None = None  # Bounding box for current frame

        # --------- Grid statistics ---------
        self.grid_rows = self.grid_cfg.get("rows", 8)
        self.grid_cols = self.grid_cfg.get("cols", 8)
        self.grid_cell_w: float = 0.0
        self.grid_cell_h: float = 0.0
        self.grid_data = {
            "dwell_time": np.zeros((self.grid_rows, self.grid_cols)),
            "visit_count": np.zeros((self.grid_rows, self.grid_cols)),
            "total_speed": np.zeros((self.grid_rows, self.grid_cols)),
            "max_speed": np.zeros((self.grid_rows, self.grid_cols)),
            "activity_time": np.zeros((self.grid_rows, self.grid_cols)),
            "transition_matrix": np.zeros((self.grid_rows * self.grid_cols,
                                           self.grid_rows * self.grid_cols))
        }
        self.last_grid_position: Tuple[int, int] | None = None

        # --------- CSV output ---------
        self.csv_data = []

        # --------- Trajectory analysis database ---------
        # Save trajectories per video to avoid cross-video interference
        self.current_video_trajectory = pd.DataFrame(columns=[
            'video_name', 'datetime', 'x', 'y', 'state', 'speed',
            'grid_row', 'grid_col', 'region', 'activity_type',
            'dwell_time', 'frame_width', 'frame_height', 'fps', 'frame_skip',
            'grid_rows', 'grid_cols'
        ])

        # --------- False-positive filter ---------
        self.false_positive_filter = {
            "min_motion_frames": self.mot_cfg.get("min_motion_frames", 3),
            "min_size_change": self.mot_cfg.get("min_size_change", 0.1),
            "color_threshold": self.mot_cfg.get("color_threshold", 0.7),
            "history": [],
            "history_size": self.mot_cfg.get("history_size", 10),
            "min_white_variation": self.mot_cfg.get("min_white_variation", 0.1),
        }

    # --------------------------------------------------------
    #                    Main entry
    # --------------------------------------------------------
    def process_video(self, vid_path: str | Path, show: bool, video_info: Optional[Dict] = None,
                      start_frame: Optional[int] = None, end_frame: Optional[int] = None,
                      custom_output_dir: Optional[str] = None) -> Dict:
        """Video File,.start_frameend_frame.
        custom_output_dir: (UI)"""
        vid_path = Path(vid_path)

        
        if custom_output_dir:
            out_dir = Path(custom_output_dir)
        else:
            # Video File
            video_name = vid_path.stem
            out_dir = Path(self.gen_cfg["output_dir"]) / video_name

        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {vid_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.video_fps = fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_width = w
        self.video_height = h

        #  ( 20250616_00_03.mp4)
        try:
            match = re.match(r'(\d{8})_(\d{2})_(\d{2})', vid_path.stem)
            if match:
                date_str, start_hour_str, end_hour_str = match.groups()
                self.video_start_time = dt.datetime.strptime(f"{date_str}{start_hour_str}0000", "%Y%m%d%H%M%S")
            else:
                raise ValueError("")
        except Exception as e:
            logger.warning(f"Unable to parse video start time: {e},using file modification time")
            self.video_start_time = dt.datetime.fromtimestamp(vid_path.stat().st_mtime)

        # ()
        if self.gen_cfg.get("save_processed_video", False):
            video_writer = cv2.VideoWriter(
                str(out_dir / f"processed_{vid_path.stem}.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )
        else:
            video_writer = None

        
        self.grid_cell_w = w / self.grid_cols
        self.grid_cell_h = h / self.grid_rows
        logger.info(f"Grid initialized: {self.grid_rows} x {self.grid_cols}, "
                    f"cell {self.grid_cell_w:.1f}x{self.grid_cell_h:.1f}px")

        
        sf = start_frame if start_frame is not None else 0
        ef = end_frame if end_frame is not None else total
        if ef > total:
            ef = total

        # Statistics
        self.absent_t = self.static_t = self.moving_t = 0.0
        self.last_video_time = None
        self.frame_total = 0
        self.frame_detect = 0
        self.background = None
        self.heatmap = None
        self.tracker = PandaMotionTracker(
            shift_thr=self.mot_cfg["bbox_shift_threshold"],
            size_thr=self.mot_cfg["bbox_size_change"],
        )
        self._prev_bbox = None
        self._current_bbox = None
        self.last_grid_position = None

        # Reset grid data
        for key in self.grid_data:
            if key == "transition_matrix":
                self.grid_data[key] = np.zeros((self.grid_rows * self.grid_cols,
                                                self.grid_rows * self.grid_cols))
            else:
                self.grid_data[key] = np.zeros((self.grid_rows, self.grid_cols))

        # Save trajectories per video to avoid cross-video interference
        self.current_video_trajectory = pd.DataFrame(columns=[
            'video_name', 'datetime', 'x', 'y', 'state', 'speed',
            'grid_row', 'grid_col', 'region', 'activity_type',
            'dwell_time', 'frame_width', 'frame_height', 'fps', 'frame_skip',
            'grid_rows', 'grid_cols'
        ])
        self.false_positive_filter["history"] = []

        logger.info(f"Video info: {total}  , {fps:.1f} FPS , {w}x{h}")
        logger.info(f"Analysis frame range: {sf} - {ef} ({ef - sf})")

        start = time.time()
        frame_idx, last_proc_idx = 0, -1

        
        import sys
        sys.stdout.reconfigure(encoding='utf-8')

        pbar = tqdm(
            total=ef - sf,
            desc=vid_path.name,
            unit="frame",
            ncols=90,
            ascii=False,
            dynamic_ncols=False,
            leave=True,
            mininterval=1.0,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        cur_frame = sf
        while cur_frame < ef:
            ok, frame = cap.read()
            if not ok:
                break
            cur_frame += 1
            frame_idx += 1

            if cur_frame % self.gen_cfg["frame_skip"]:
                pbar.update(1)
                continue

            video_time = cur_frame / fps
            delta_frames = cur_frame - (last_proc_idx if last_proc_idx >= 0 else 0)
            last_proc_idx = cur_frame

            if self.background is None:
                self.background = frame.copy()
                self.heatmap = np.zeros((h, w), dtype=np.uint32)

            # -------  -------
            state = self._process_frame(frame, delta_frames, fps, video_time)

            # -------  -------
            disp = frame.copy()

            
            if self.grid_cfg.get("show_grid", True):
                self._draw_grid(disp)

            # Bounding box for current frame
            if self._current_bbox is not None:
                x1, y1, x2, y2 = map(int, self._current_bbox)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.gen_cfg["show_heatmap"]:
                disp = self._overlay_heatmap(disp)

            
            if video_writer is not None:
                video_writer.write(disp)

            if show:
                cv2.imshow("Panda Monitor", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)

        cap.release()
        if video_writer is not None:
            video_writer.release()

        # ,  OpenCV  GUI 
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error as e:
                logger.warning(f"cv2.destroyAllWindows() ,:{e}")

        pbar.close()

        # -------  -------
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_path = None
        if self.gen_cfg.get("save_trajectory_plot", True):
            traj_path = out_dir / f"trajectory_{vid_path.stem}_{now}.png"
            self.tracker.save_trajectory_plot(traj_path, background=self.background)
            logger.info(f"Trajectory plot saved: {traj_path}")

        heat_path = out_dir / f"heatmap_{vid_path.stem}_{now}.png"
        self._save_heatmap_image(heat_path)
        logger.info(f"Heatmap saved: {heat_path}")

        # Trajectory
        trajectory_analysis = self._analyze_trajectory(vid_path.name)

        # TrajectoryExcel - :video_name()
        traj_db_path = self._save_trajectory_database(out_dir, vid_path.stem)

        # Grid analysis report - :out_dir
        grid_analysis_path = self._save_grid_analysis(out_dir, vid_path.stem)

        
        tot = max(self.absent_t + self.static_t + self.moving_t, 1e-6)
        mot = self.tracker.generate_motion_report()

        result = {
            "video_path": str(vid_path),
            "video_name": vid_path.name,
            "total_frames": self.frame_total,
            "detected_frames": self.frame_detect,
            "detection_rate": self.frame_detect / self.frame_total * 100 if self.frame_total > 0 else 0,
            "total_time": tot,
            "absent_time": self.absent_t,
            "static_time": self.static_t,
            "moving_time": self.moving_t,
            "absent_percent": self.absent_t / tot * 100,
            "static_percent": self.static_t / tot * 100,
            "moving_percent": self.moving_t / tot * 100,
            "total_distance": mot.get('total_distance', 0),
            "avg_speed": mot.get('avg_speed', 0),
            "max_speed": mot.get('max_speed', 0),
            "movement_count": mot.get('movement_count', 0),
            "fps": fps,
            "frame_skip": self.gen_cfg["frame_skip"],
            "analysis_time": time.time() - start,
            # Trajectory
            "top_regions": trajectory_analysis.get('top_regions', []),
            "frequent_paths": trajectory_analysis.get('frequent_paths', []),
            "speed_by_region": trajectory_analysis.get('speed_by_region', []),
            "activity_by_hour": trajectory_analysis.get('activity_by_hour', []),
            "grid_analysis": self._get_grid_analysis(),
            # Path
            "trajectory_path": str(traj_path) if traj_path else "",
            "heatmap_path": str(heat_path),
            "grid_analysis_path": str(grid_analysis_path) if grid_analysis_path else "",
            "trajectory_db_path": str(traj_db_path) if traj_db_path else "",
            "processed_video_path": str(out_dir / f"processed_{vid_path.stem}.mp4") if video_writer else "",
            "output_dir": str(out_dir)  
        }

        # Video info
        if video_info:
            result.update(video_info)

        self.csv_data.append(result)
        logger.info(f" {vid_path.name} !")
        logger.info(f": {out_dir}")

        
        self.db_manager.add_analysis_result(result)

        return result

    def process_video_directory(self, video_dir: str, date_range: str = None, time_range: str = None,
                                show: bool = False, output_base_dir: Optional[str] = None) -> None:
        """Video File,Folder"""
        video_dir = Path(video_dir)
        if not video_dir.is_dir():
            raise ValueError(f": {video_dir}")

        # Video File
        video_files = list(video_dir.rglob("*.mp4"))  # mp4

        # ,
        if date_range and time_range:
            
            start_dt, end_dt = self._parse_datetime_range(date_range, time_range)
            filtered_videos = []
            for vf in video_files:
                try:
                    v_start, v_end = self._get_video_file_time_range(vf)
                    
                    if not (v_end <= start_dt or v_start >= end_dt):
                        filtered_videos.append(vf)
                except ValueError:
                    continue
            video_files = filtered_videos

        logger.info(f" {len(video_files)} Video File,Start Analysis...")

        for vf in video_files:
            logger.info(f": {vf.name}")
            try:
                # Video info()
                video_info = {}
                if date_range and time_range:
                    v_start, v_end = self._get_video_file_time_range(vf)
                    video_info = {
                        "start_time": v_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "end_time": v_end.strftime("%Y-%m-%d %H:%M:%S")
                    }

                # Path,Folder
                relative_path = vf.relative_to(video_dir)

                # ,Folder
                if output_base_dir:
                    # ,
                    output_subdir = Path(output_base_dir) / relative_path.parent
                    custom_output_dir = str(output_subdir / vf.stem)
                else:
                    custom_output_dir = None

                self.process_video(vf, show=show, video_info=video_info, custom_output_dir=custom_output_dir)
            except Exception as e:
                logger.error(f" {vf} : {e}")

    def _draw_grid(self, frame: np.ndarray) -> None:
        """"""
        h, w = frame.shape[:2]

        
        for i in range(1, self.grid_rows):
            y = int(i * self.grid_cell_h)
            cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)

        
        for j in range(1, self.grid_cols):
            x = int(j * self.grid_cell_w)
            cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)

        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                x = int(j * self.grid_cell_w + 5)
                y = int((i + 1) * self.grid_cell_h - 5)
                cv2.putText(frame, f"R{i}C{j}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # --------------------------------------------------------
    #                    Trajectory
    # --------------------------------------------------------
    def _analyze_trajectory(self, video_name: str) -> Dict:
        """Trajectory,"""
        if self.current_video_trajectory.empty:
            return {}

        
        video_db = self.current_video_trajectory
        if video_db.empty:
            return {}

        # 1. (DwellTime)
        region_dwell = video_db.groupby('region')['dwell_time'].sum().reset_index()
        region_dwell = region_dwell.sort_values('dwell_time', ascending=False)
        top_regions = region_dwell.head(10).to_dict('records')

        # 2. ()
        video_db['next_region'] = video_db['region'].shift(-1)
        path_counts = video_db.groupby(['region', 'next_region']).size().reset_index(name='count')
        path_counts = path_counts.sort_values('count', ascending=False)
        frequent_paths = path_counts.head(10).to_dict('records')

        # 3. SpeedAnalysis
        region_speed = video_db.groupby('region')['speed'].agg(['mean', 'max']).reset_index()
        region_speed.columns = ['region', 'avg_speed', 'max_speed']
        region_speed = region_speed.sort_values('avg_speed', ascending=False)
        speed_by_region = region_speed.to_dict('records')

        # 4. 
        video_db['hour'] = video_db['datetime'].dt.hour
        hour_activity = video_db.groupby('hour')['state'].apply(
            lambda x: (x == 'moving').sum() / len(x) * 100
        ).reset_index(name='activity_percent')
        activity_by_hour = hour_activity.to_dict('records')

        return {
            'top_regions': top_regions,
            'frequent_paths': frequent_paths,
            'speed_by_region': speed_by_region,
            'activity_by_hour': activity_by_hour
        }

    def _get_grid_analysis(self) -> Dict:
        """"""
        grid_analysis = {}

        # 1. DwellTime
        dwell_time = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                dwell_minutes = self.grid_data["dwell_time"][i, j] * self.gen_cfg["frame_skip"] / self.video_fps / 60
                dwell_time.append({
                    "grid_row": i,
                    "grid_col": j,
                    "region": f"R{i}C{j}",
                    "dwell_minutes": dwell_minutes,
                    "dwell_percent": dwell_minutes / (self.static_t / 60) * 100 if self.static_t > 0 else 0
                })
        dwell_time_sorted = sorted(dwell_time, key=lambda x: x["dwell_minutes"], reverse=True)
        grid_analysis["dwell_time"] = dwell_time_sorted[:10]  # 10

        # 2. VisitCount
        visit_count = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                count = self.grid_data["visit_count"][i, j]
                visit_count.append({
                    "grid_row": i,
                    "grid_col": j,
                    "region": f"R{i}C{j}",
                    "visit_count": count
                })
        visit_count_sorted = sorted(visit_count, key=lambda x: x["visit_count"], reverse=True)
        grid_analysis["visit_count"] = visit_count_sorted[:10]

        # 3. SpeedAnalysis
        speed_analysis = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                total_speed = self.grid_data["total_speed"][i, j]
                visit_count = self.grid_data["visit_count"][i, j]
                avg_speed = total_speed / visit_count if visit_count > 0 else 0
                max_speed = self.grid_data["max_speed"][i, j]
                speed_analysis.append({
                    "grid_row": i,
                    "grid_col": j,
                    "region": f"R{i}C{j}",
                    "avg_speed": avg_speed,
                    "max_speed": max_speed
                })
        speed_analysis_sorted = sorted(speed_analysis, key=lambda x: x["avg_speed"], reverse=True)
        grid_analysis["speed_analysis"] = speed_analysis_sorted

        # 4. ActivityTime
        activity_time = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                activity_minutes = self.grid_data["activity_time"][i, j] * self.gen_cfg[
                    "frame_skip"] / self.video_fps / 60
                activity_time.append({
                    "grid_row": i,
                    "grid_col": j,
                    "region": f"R{i}C{j}",
                    "activity_minutes": activity_minutes,
                    "activity_percent": activity_minutes / (self.moving_t / 60) * 100 if self.moving_t > 0 else 0
                })
        activity_time_sorted = sorted(activity_time, key=lambda x: x["activity_minutes"], reverse=True)
        grid_analysis["activity_time"] = activity_time_sorted[:10]

        # 5. FrequentPaths
        frequent_paths = []
        for i in range(self.grid_rows * self.grid_cols):
            for j in range(self.grid_rows * self.grid_cols):
                count = self.grid_data["transition_matrix"][i, j]
                if count > 0:
                    start_row = i // self.grid_cols
                    start_col = i % self.grid_cols
                    end_row = j // self.grid_cols
                    end_col = j % self.grid_cols
                    frequent_paths.append({
                        "start_region": f"R{start_row}C{start_col}",
                        "end_region": f"R{end_row}C{end_col}",
                        "path": f"R{start_row}C{start_col}->R{end_row}C{end_col}",
                        "transition_count": count
                    })
        frequent_paths_sorted = sorted(frequent_paths, key=lambda x: x["transition_count"], reverse=True)
        grid_analysis["frequent_paths"] = frequent_paths_sorted[:10]

        return grid_analysis

    def _save_trajectory_database(self, out_dir: Path, video_name: str) -> Path:
        """Save trajectory data Excel with selected columns only."""
        if self.current_video_trajectory.empty:
            logger.warning("Trajectory database is empty; cannot save")
            return None

        video_db = self.current_video_trajectory.copy()

        drop_cols = [
            "speed",
            "activity_type",
            "dwell_time",
            "next_region"
        ]
        video_db = video_db.drop(
            columns=[col for col in drop_cols if col in video_db.columns],
            errors="ignore"
        )

        if "datetime" in video_db.columns:
            video_db["datetime"] = pd.to_datetime(video_db["datetime"]).dt.strftime("%Y%m%d_%H_%M_%S")

        excel_path = out_dir / f"trajectory_{video_name}.xlsx"
        try:
            video_db.to_excel(excel_path, index=False)
            logger.info(f"Trajectory database saved as Excel: {excel_path}")
            return excel_path
        except Exception as e:
            logger.error(f"Error saving trajectory database: {e}")
            return None

    def _save_grid_analysis(self, out_dir: Path, video_name: str) -> Path:
        """Save grid analysis Excel with selected sheets and columns only."""
        grid_analysis = self._get_grid_analysis()
        excel_path = out_dir / f"grid_analysis_{video_name}.xlsx"

        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                if grid_analysis.get("visit_count"):
                    pd.DataFrame(grid_analysis["visit_count"]).to_excel(
                        writer, index=False, sheet_name="VisitCount"
                    )

                if grid_analysis.get("frequent_paths"):
                    pd.DataFrame(grid_analysis["frequent_paths"]).to_excel(
                        writer, index=False, sheet_name="FrequentPaths"
                    )

                grid_summary = []
                for i in range(self.grid_rows):
                    for j in range(self.grid_cols):
                        grid_summary.append({
                            "grid_row": i,
                            "grid_col": j,
                            "region": f"R{i}C{j}",
                            "visit_count": self.grid_data["visit_count"][i, j]
                        })

                pd.DataFrame(grid_summary).to_excel(
                    writer, index=False, sheet_name="GridSummary"
                )

            logger.info(f"Grid analysis report saved as Excel: {excel_path}")
            return excel_path
        except Exception as e:
            logger.error(f"Error saving grid analysis report: {e}")
            return None

    # --------------------------------------------------------
    #                    Per-frame core
    # --------------------------------------------------------
    def _process_frame(
            self, frame: np.ndarray, d_frames: int, fps: float, vtime: float
    ) -> str:
        frame_det, scale = self._resize_keep_ratio(frame)
        found, bbox = self._detect_panda(frame_det, scale)

        delta_t = 0.0 if self.last_video_time is None else vtime - self.last_video_time
        self.last_video_time = vtime

        
        timestamp = self.video_start_time + dt.timedelta(seconds=vtime)

        # :()
        self._current_bbox = None  
        if found and bbox is not None:
            is_false_positive = self._is_false_positive(frame, bbox)
            if is_false_positive:
                found = False
                bbox = None
            else:
                self._current_bbox = bbox  

        if not found or bbox is None:
            self.absent_t += delta_t
            self.tracker.update(None, vtime, False)
            self._prev_bbox = None
            self.last_grid_position = None
            # Trajectory()
            self._record_trajectory_point(None, None, 'absent', 0, timestamp, frame)
            return "absent"

        # =======  =======
        self._update_heatmap(bbox)

        moving_pixel = (
            self._detect_motion(frame, bbox) if self.gen_cfg["enable_motion_detection"] else False
        )
        moving_bbox = self.tracker.bbox_shift_moving(bbox, self._prev_bbox)
        self._prev_bbox = bbox.copy()
        moving = moving_pixel or moving_bbox

        
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        # (px/)
        speed = self.tracker.get_current_speed()

        # Trajectory
        self._record_trajectory_point(cx, cy, 'moving' if moving else 'static', speed, timestamp, frame)

        
        if cx is not None and cy is not None:
            
            grid_col = min(int(cx / self.grid_cell_w), self.grid_cols - 1)
            grid_row = min(int(cy / self.grid_cell_h), self.grid_rows - 1)

            
            self.grid_data["visit_count"][grid_row, grid_col] += 1
            self.grid_data["total_speed"][grid_row, grid_col] += speed
            if speed > self.grid_data["max_speed"][grid_row, grid_col]:
                self.grid_data["max_speed"][grid_row, grid_col] = speed

            if moving:
                self.grid_data["activity_time"][grid_row, grid_col] += delta_t
            else:
                self.grid_data["dwell_time"][grid_row, grid_col] += delta_t

            
            if self.last_grid_position is not None:
                last_index = self.last_grid_position[0] * self.grid_cols + self.last_grid_position[1]
                current_index = grid_row * self.grid_cols + grid_col
                if last_index != current_index:
                    self.grid_data["transition_matrix"][last_index, current_index] += 1

            self.last_grid_position = (grid_row, grid_col)

        if moving:
            self.moving_t += delta_t
            state = "moving"
        else:
            self.static_t += delta_t
            state = "static"

        self.tracker.update(bbox, vtime, moving)
        return state

    def _is_false_positive(self, frame: np.ndarray, bbox: np.ndarray) -> bool:
        """Check if detection is a false positive (e.g., white rock)"""
        # 1. 
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ()
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 30, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # px
        white_ratio = cv2.countNonZero(white_mask) / (roi.size / 3)

        # 3. :
        # ,
        std_dev = np.std(roi, axis=(0, 1))
        avg_std = np.mean(std_dev)

        # 4. :
        motion_history = []
        if self._prev_bbox is not None:
            
            prev_area = (self._prev_bbox[2] - self._prev_bbox[0]) * (self._prev_bbox[3] - self._prev_bbox[1])
            curr_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            size_change = abs(curr_area - prev_area) / prev_area

            
            prev_center = np.array([(self._prev_bbox[0] + self._prev_bbox[2]) / 2,
                                    (self._prev_bbox[1] + self._prev_bbox[3]) / 2])
            curr_center = np.array([(bbox[0] + bbox[2]) / 2,
                                    (bbox[1] + bbox[3]) / 2])
            shift_distance = np.linalg.norm(curr_center - prev_center)

            
            motion_history.append({
                'size_change': size_change,
                'shift_distance': shift_distance
            })

            # ,
            if shift_distance < 2 and size_change < self.false_positive_filter["min_size_change"]:
                
                self.false_positive_filter["history"].append(1)
                if len(self.false_positive_filter["history"]) > self.false_positive_filter["history_size"]:
                    self.false_positive_filter["history"].pop(0)
            else:
                # ,
                self.false_positive_filter["history"] = []

        # : +  + 
        if (white_ratio > self.false_positive_filter["color_threshold"] and
                avg_std < 15 and
                sum(self.false_positive_filter["history"]) >= self.false_positive_filter["min_motion_frames"]):
            logger.debug(f": ={white_ratio:.2f}, ={avg_std:.2f}")
            return True

        return False

    def _record_trajectory_point(self, x: int, y: int, state: str, speed: float,
                                 timestamp: dt.datetime, frame: np.ndarray) -> None:
        """Trajectory"""
        
        grid_row = grid_col = None
        region = "unknown"

        if x is not None and y is not None:
            grid_col = min(int(x / self.grid_cell_w), self.grid_cols - 1)
            grid_row = min(int(y / self.grid_cell_h), self.grid_rows - 1)
            region = f"R{grid_row}C{grid_col}"

        # Type()
        activity_type = "resting" if state == "static" else "moving"
        if region.startswith("R0") and state == "static":
            activity_type = "feeding"
        elif region.startswith("R2") and state == "static":
            activity_type = "sleeping"

        # DwellTime()
        dwell_time = 0
        if state == "static" and not self.current_video_trajectory.empty:
            last_point = self.current_video_trajectory.iloc[-1]
            if (last_point['state'] == 'static' and
                    last_point['grid_row'] == grid_row and
                    last_point['grid_col'] == grid_col):
                dwell_time = (timestamp - last_point['datetime']).total_seconds()

        new_point = {
            'video_name': Path(self.cfg["general"]["video_path"]).stem if "video_path" in self.cfg["general"] else "",
            'datetime': timestamp,
            'x': x,
            'y': y,
            'state': state,
            'speed': speed,
            'grid_row': grid_row,
            'grid_col': grid_col,
            'region': region,
            'activity_type': activity_type,
            'dwell_time': dwell_time,
            # :
            'frame_width': self.video_width,
            'frame_height': self.video_height,
            'fps': self.video_fps,
            'frame_skip': self.gen_cfg["frame_skip"],
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'datetime': timestamp  
        }

        new_df = pd.DataFrame([new_point])[self.current_video_trajectory.columns]

        #   DataFrame  concat
        if self.current_video_trajectory.empty:
            # : new_df 
            self.current_video_trajectory = new_df.copy()
        else:
            #  concat
            self.current_video_trajectory = pd.concat(
                [self.current_video_trajectory, new_df],
                ignore_index=True
            )

    # --------------------------------------------------------
    #                     YOLO
    # --------------------------------------------------------
    def _get_panda_cls_id(self) -> int:
        for i, n in self.model.names.items():
            if n.lower() == "panda":
                return i
        logger.warning(" panda , 0")
        return 0

    def _resize_keep_ratio(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        tw, th = self.det_cfg["process_width"], self.det_cfg["process_height"]
        scale = min(tw / w, th / h)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img, scale

    def _detect_panda(self, img: np.ndarray, scale: float) -> Tuple[bool, np.ndarray | None]:
        self.frame_total += 1
        rs = self.model(
            img, conf=self.det_cfg["conf_threshold"], verbose=False, device=self.device
        )
        for r in rs:
            for b in r.boxes:
                if int(b.cls[0]) == self.panda_cls_id:
                    self.frame_detect += 1
                    box = b.xyxy[0].cpu().numpy()
                    if scale != 1:
                        box = box / scale
                    return True, box
        return False, None

    # --------------------------------------------------------
    #                 px
    # --------------------------------------------------------
    def _detect_motion(self, frame: np.ndarray, bbox: np.ndarray) -> bool:
        mc = self.mot_cfg
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (mc["blur_size"],) * 2, 0)
        mask = cv2.inRange(gray, mc["dark_threshold"], mc["bright_threshold"])

        if not hasattr(self, "_prev_roi") or self._prev_roi is None:
            self._prev_roi = gray
            return False

        if self._prev_roi.shape != gray.shape:
            self._prev_roi = cv2.resize(self._prev_roi, (gray.shape[1], gray.shape[0]))

        diff = cv2.absdiff(gray, self._prev_roi)
        self._prev_roi = gray.copy()

        _, m = cv2.threshold(
            diff, int(mc["motion_threshold"] * 255), 255, cv2.THRESH_BINARY
        )
        comb = cv2.bitwise_and(mask, m)

        cnts, _ = cv2.findContours(comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mv_area = sum(
            cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > mc["contour_area_threshold"]
        )
        ratio = mv_area / gray.size

        if not hasattr(self, "_state_buf"):
            self._state_buf = []
        self._state_buf.append(1 if ratio > mc["motion_detection_threshold"] else 0)
        if len(self._state_buf) > mc["state_buffer_size"]:
            self._state_buf.pop(0)

        return sum(self._state_buf) >= mc["min_motion_frames"]

    # --------------------------------------------------------
    #                    Heatmap & 
    # --------------------------------------------------------
    def _update_heatmap(self, bbox: np.ndarray) -> None:
        if self.heatmap is None:
            return
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.heatmap.shape[1] - 1, x2), min(self.heatmap.shape[0] - 1, y2)
        self.heatmap[y1:y2, x1:x2] += 1

    def _overlay_heatmap(self, frame: np.ndarray) -> np.ndarray:
        if self.heatmap is None or np.max(self.heatmap) == 0:
            return frame
        heat = self.heatmap.astype(np.float32)
        norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1.0, color, self.gen_cfg["heatmap_alpha"], 0)

    # --------------------------------------------------------
    #                   Heatmap()
    # --------------------------------------------------------
    def _save_heatmap_image(self, path: Path) -> None:
        """
        Faded grayscale background + 
        DwellTime()
        """
        if self.heatmap is None or np.max(self.heatmap) == 0:
            logger.warning("Heatmap,")
            return

        # ----------  ----------
        bg_cfg = self.gen_cfg.get("heatmap_background", "auto")
        if isinstance(bg_cfg, str) and bg_cfg.lower() != "auto":
            bg_img = cv2.imread(bg_cfg, cv2.IMREAD_COLOR)
            if bg_img is None:
                logger.warning(f" {bg_cfg},")
                bg_img = self.background.copy()
        else:
            bg_img = self.background.copy()
        if bg_img.shape[:2] != self.heatmap.shape:
            bg_img = cv2.resize(bg_img, (self.heatmap.shape[1], self.heatmap.shape[0]))

        # ---------- A. Faded grayscale background ----------
        bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        # ---------- A. Grayscale fading background (more "transparent/whitening" effect) ----------
        bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)

        # NEW: wash-out (blend with white)
        white = np.full_like(bg_gray, 255)
        bg_keep = 0.18  # Background retention ratio: The smaller, the lighter (0.12~0.25 is acceptable)
        faded = cv2.addWeighted(bg_gray, bg_keep, white, 1.0 - bg_keep, 0)

        # ---------- B.  ----------
        heat = self.heatmap.astype(np.float32)
        mask_pos = (heat > 0).astype(np.uint8)
        norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm = cv2.bitwise_and(norm, norm, mask=mask_pos)
        red_layer = np.zeros_like(faded)
        red_layer[..., 2] = norm
        overlay = cv2.addWeighted(faded, 1.0, red_layer, 0.90, 0)

        # ---------- C.  ----------
        topk = int(self.gen_cfg.get("topk_hotspots", 3))
        grid_points = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                
                cx = int(j * self.grid_cell_w + self.grid_cell_w / 2)
                cy = int(i * self.grid_cell_h + self.grid_cell_h / 2)
                cx = max(0, min(cx, overlay.shape[1] - 1))
                cy = max(0, min(cy, overlay.shape[0] - 1))

                # DwellTime()
                dwell_seconds = self.grid_data["dwell_time"][i, j]
                dwell_min = dwell_seconds * self.gen_cfg["frame_skip"] / self.video_fps / 60.0
                if dwell_min > 0.01:  # DwellTime
                    grid_points.append((i, j, cx, cy, dwell_min))

        # DwellTime,topk
        grid_points.sort(key=lambda x: x[4], reverse=True)
        raw_topk = grid_points[:topk]

        # ---------- D. DwellTime() ----------
        # if raw_topk:
        #    H, W = overlay.shape[:2]
        #   placed = []  # [(x0,y0,x1,y1)]
        #    for r, c, cx, cy, dwell_min in raw_topk:
        #       txt = f"{dwell_min:.1f}min"
        #        scale, thick = 0.6, 2
        #        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)

        
        #        cand = [(cx + 5, cy + th), (cx + 5, cy - 5),
        #                (cx - tw - 5, cy + th), (cx - tw - 5, cy - 5)]
        #        found_pos = None
        #        for x0, y_base in cand:
        #            y_top = y_base - th
        #            rect = (x0, y_top, x0 + tw, y_base + bl)
        #            inside = 0 <= x0 and 0 <= y_top and rect[2] <= W and rect[3] <= H
        #            overlap = any(not (rect[2] < px0 or rect[0] > px1 or rect[3] < py0 or rect[1] > py1)
        #                          for px0, py0, px1, py1 in placed)
        #            if inside and not overlap:
        #                found_pos = (x0, y_base, rect)
        #                break
        #        if found_pos is None:  # ,
        #           x0 = min(max(0, cx), W - tw)
        #            y_base = min(max(th, cy), H - bl)
        #            rect = (x0, y_base - th, x0 + tw, y_base + bl)
        #           found_pos = (x0, y_base, rect)
        #       x0, y_base, rect = found_pos
        #       placed.append(rect)

        #  + 
        #       cv2.putText(overlay, txt, (x0, y_base),
        #                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        #        cv2.putText(overlay, txt, (x0, y_base),
        #                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

        cv2.imwrite(str(path), overlay)
        logger.info(f"Heatmap: {path.name}")

        # ---------- E.  ----------
        if not self.gen_cfg.get("raw_heatmap", True):
            return
        raw_path = path.with_name(path.stem + "_raw.png")
        import matplotlib.pyplot as plt
        from matplotlib import ticker
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        im = ax.imshow(self.heatmap, cmap="jet", origin="upper")
        ax.axis("off")

        cb_ax = inset_axes(ax, width="3%", height="100%", loc="lower left",
                           bbox_to_anchor=(1.02, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = fig.colorbar(im, cax=cb_ax)

        def _fmt(v, _):
            sec = v * self.gen_cfg["frame_skip"] / self.video_fps
            return f"{sec / 60:.1f}"

        cbar.formatter = ticker.FuncFormatter(_fmt)
        cbar.set_label("Dwell time (min)")
        cbar.update_ticks()

        fig.savefig(str(raw_path), transparent=True, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Clean heatmap saved: {raw_path}")

    # --------------------------------------------------------
    
    # --------------------------------------------------------
    def _write_report(
            self, path: Path, traj_path: Path | None, heat_path: Path | None
    ) -> None:
        tot = max(self.absent_t + self.static_t + self.moving_t, 1e-6)
        mot = self.tracker.generate_motion_report()
        txt = f"""
: {dt.datetime.now():%Y-%m-%d %H:%M:%S}

:
              : {self.frame_total}
     : {self.frame_detect} ({self.frame_detect / self.frame_total * 100:.2f}%)

:
              : {tot:.2f}s
            : {self.absent_t:.2f}s ({self.absent_t / tot * 100:.2f}%)
            : {self.static_t:.2f}s ({self.static_t / tot * 100:.2f}%)
            : {self.moving_t:.2f}s ({self.moving_t / tot * 100:.2f}%)

:
              : {mot['total_distance']:.2f} m
            : {mot['avg_speed']:.2f} m/s
            : {mot['max_speed']:.2f} m/s
          : {mot['movement_count']}

:
            : {self.grid_rows} x {self.grid_cols}
  cell        : {self.grid_cell_w:.1f}x{self.grid_cell_h:.1f}px

:
"""
        if traj_path:
            txt += f"  Trajectory         : {traj_path.name}\n"
        if heat_path:
            txt += f"  Heatmap       : {heat_path.name}\n"

        path.write_text(txt, encoding="utf-8")
        logger.info(f" {path}")

    def _get_video_fps(self, video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        return fps

    def _parse_datetime_range(self, date_range: str, time_range: str) -> tuple[dt.datetime, dt.datetime]:
        """--date--time"""
        # date_range: 20250602_20250603
        # time_range: 23_04()
        start_date_str, end_date_str = date_range.split('_')
        start_time_str, end_time_str = time_range.split('_')
        start_date = dt.datetime.strptime(start_date_str, '%Y%m%d')
        end_date = dt.datetime.strptime(end_date_str, '%Y%m%d')
        start_hour = int(start_time_str)
        end_hour = int(end_time_str)
        
        start_dt = start_date.replace(hour=start_hour, minute=0, second=0)
        # ,,
        if end_hour > start_hour or (end_hour == start_hour and start_date == end_date):
            end_dt = start_date.replace(hour=end_hour, minute=0, second=0)
        else:
            end_dt = (start_date + dt.timedelta(days=1)).replace(hour=end_hour, minute=0, second=0)
        # date,end_date+end_hour
        if end_date > start_date:
            end_dt = end_date.replace(hour=end_hour, minute=0, second=0)
        return start_dt, end_dt

    def _get_video_file_time_range(self, video_file: Path) -> tuple[dt.datetime, dt.datetime]:
        # : YYYYMMDD_startHH_endHH.mp4
        match = re.match(r"(\d{8})_(\d{2})_(\d{2})\.mp4", video_file.name)
        if not match:
            raise ValueError(f": {video_file.name}")
        date_str, start_h, end_h = match.groups()
        date = dt.datetime.strptime(date_str, '%Y%m%d')
        start_dt = date.replace(hour=int(start_h), minute=0, second=0)
        if int(end_h) == 24:
            end_dt = (date + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0)
        else:
            end_dt = date.replace(hour=int(end_h), minute=0, second=0)
            # end_h < start_h,
            if int(end_h) < int(start_h):
                end_dt += dt.timedelta(days=1)
        return start_dt, end_dt

    def _generate_night_segments(self, date_range: str, time_range: str) -> list[tuple[dt.datetime, dt.datetime]]:
        """date_rangetime_range(datetime)"""
        start_date_str, end_date_str = date_range.split('_')
        start_time_str, end_time_str = time_range.split('_')
        start_date = dt.datetime.strptime(start_date_str, '%Y%m%d')
        end_date = dt.datetime.strptime(end_date_str, '%Y%m%d')
        start_hour = int(start_time_str)
        end_hour = int(end_time_str)
        segments = []
        cur_date = start_date
        while cur_date <= end_date:
            seg_start = cur_date.replace(hour=start_hour, minute=0, second=0)
            if end_hour < start_hour:
                
                seg_end = (cur_date + dt.timedelta(days=1)).replace(hour=end_hour, minute=0, second=0)
            else:
                seg_end = cur_date.replace(hour=end_hour, minute=0, second=0)
            segments.append((seg_start, seg_end))
            cur_date += dt.timedelta(days=1)
        return segments


# ============================================================
#                         CLI
# ============================================================
def load_config(path: str | Path | None) -> Dict:
    default = Path(__file__).with_name("config.yaml")
    cfg_path = Path(path or default)
    if not cfg_path.exists():
        raise FileNotFoundError(f": {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    
    def _g(sec, k, v):
        data.setdefault(sec, {})
        data[sec].setdefault(k, v)

    
    _g("general", "frame_skip", 2)
    _g("general", "enable_motion_detection", True)
    _g("general", "show_heatmap", False)
    _g("general", "save_trajectory_plot", True)
    _g("general", "heatmap_alpha", 0.5)
    _g("general", "heatmap_background", "auto")
    _g("general", "topk_hotspots", 3)
    _g("general", "raw_heatmap", True)
    _g("general", "grid_init_frames", 30)
    _g("general", "database_path", "panda_analysis_database.xlsx")  # Unified database path
    _g("general", "output_dir", "results")  

    
    _g("detection", "conf_threshold", 0.40)
    _g("detection", "process_width", 640)
    _g("detection", "process_height", 480)

    
    _g("motion", "dark_threshold", 15)
    _g("motion", "bright_threshold", 245)
    _g("motion", "motion_threshold", 0.012)
    _g("motion", "contour_area_threshold", 8)
    _g("motion", "blur_size", 5)
    _g("motion", "state_buffer_size", 5)
    _g("motion", "min_motion_frames", 3)
    _g("motion", "motion_detection_threshold", 0.0004)
    _g("motion", "bbox_shift_threshold", 0.02)
    _g("motion", "bbox_size_change", 0.08)

    
    _g("grid_analysis", "rows", 8)
    _g("grid_analysis", "cols", 8)
    _g("grid_analysis", "show_grid", True)

    return data


def main() -> None:
    ap = argparse.ArgumentParser("Enhanced Panda Monitor")
    ap.add_argument("--video", help="Video File")
    ap.add_argument("--video_dir", help="")
    ap.add_argument("--model", required=True, help="YOLOv8 ")
    ap.add_argument("--config", help="Path")
    ap.add_argument("--display", action="store_true", help="()")
    ap.add_argument("--date", help=",:startYYYYMMDD_endYYYYMMDD,:20250602_20250608")
    ap.add_argument("--time", help=",:startHH_endHH,:08_09  22_08()")
    ap.add_argument("--output", help="(UI)")
    args = ap.parse_args()

    
    if not args.video and not args.video_dir:
        ap.error(" --video  --video_dir ")
    if args.video and args.video_dir:
        ap.error(" --video  --video_dir ")
    if args.time and not args.video_dir:
        ap.error("--time  --video_dir ")

    cfg = load_config(args.config)

    # ,
    if args.output:
        cfg["general"]["output_dir"] = args.output

    # Path
    if args.video:
        cfg["general"]["video_path"] = args.video
    monitor = EnhancedPandaMonitor(args.model, cfg)

    if args.video:
        
        monitor.process_video(args.video, show=args.display)
    else:
        
        monitor.process_video_directory(
            video_dir=args.video_dir,
            date_range=args.date,
            time_range=args.time,
            show=args.display,
            output_base_dir=args.output
        )


if __name__ == "__main__":
    main()