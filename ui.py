import sys
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDateTimeEdit, QGroupBox, QFileDialog,
    QFormLayout, QSplitter, QFrame, QTabWidget,
    QLineEdit, QCheckBox, QProgressBar, QTextEdit, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QHeaderView
)
from PyQt5.QtCore import QDateTime, Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QTextCursor, QIntValidator, QDesktopServices
import random
import subprocess
import yaml
import tempfile
import re
from pathlib import Path
from matplotlib.lines import Line2D
import torch
[]
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ------------------------- Analysis Worker Thread -------------------------
class AnalysisWorker(QThread):
    update_progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    analysis_finished = pyqtSignal(str)
    analysis_error = pyqtSignal(str)
    video_progress = pyqtSignal(str, int)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.cancel_flag = False

    def run(self):
        try:
            all_video_items = self.config.get('video_items', [])
            output_dir = self.config['output_dir']
            model_path = self.config['model_path']
            show_display = self.config['show_display']
            frame_skip = int(self.config['frame_skip'])

            # Build a temporary config file
            temp_cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.safe_dump({
                'general': {
                    'output_dir': output_dir,
                    'frame_skip': frame_skip,
                    'show_heatmap': False,
                    'save_trajectory_plot': True,
                    'database_path': 'panda_analysis_database.xlsx'
                },
                'detection': {'conf_threshold': 0.4, 'process_width': 416, 'process_height': 416},
                'motion': {'bbox_shift_threshold': 0.02, 'bbox_size_change': 0.08},
                'grid_analysis': {'rows': 8, 'cols': 8, 'show_grid': True}
            }, temp_cfg, sort_keys=False, allow_unicode=True)
            temp_cfg.close()

            # Collect all video files
            video_files = []
            for item in all_video_items:
                if isinstance(item, dict):  # folder
                    folder_path = item['path']
                    folder_name = item['name']
                    for mp4_file in Path(folder_path).rglob("*.mp4"):
                        video_files.append({
                            'path': mp4_file,
                            'is_folder': True,
                            'folder_name': folder_name,
                            'relative_path': mp4_file.relative_to(folder_path)
                        })
                else:  # single file
                    video_files.append({
                        'path': Path(item),
                        'is_folder': False,
                        'folder_name': Path(item).stem,
                        'relative_path': Path(item).name
                    })

            if not video_files:
                self.log_message.emit("No usable mp4 files found. Please check the selected paths.")
                return

            total = len(video_files)
            self.log_message.emit(f"Found {total} video file(s). Start analysis...")

            for idx, video_info in enumerate(video_files, 1):
                if self.cancel_flag:
                    break

                vf = video_info['path']
                is_folder = video_info['is_folder']
                folder_name = video_info['folder_name']
                relative_path = video_info['relative_path']

                # Create output directory
                if is_folder:
                    folder_output_dir = Path(output_dir) / folder_name
                    if str(relative_path) != str(relative_path.name):
                        sub_dir = str(relative_path.parent).replace('/', '_').replace('\\', '_')
                        folder_output_dir = folder_output_dir / sub_dir
                    folder_output_dir.mkdir(parents=True, exist_ok=True)
                    display_name = f"{folder_name}/{relative_path}"
                else:
                    folder_output_dir = Path(output_dir)
                    folder_output_dir.mkdir(parents=True, exist_ok=True)
                    display_name = str(relative_path)

                self.update_progress.emit(int((idx - 1) / total * 100),
                                          f"[{idx}/{total}] Start: {display_name}")
                self.video_progress.emit(display_name, int((idx - 1) / total * 100))
                self.log_message.emit(f"----- Start {display_name} -----")
                self._run_one_video(vf, model_path, temp_cfg.name, show_display, folder_output_dir)
                self.log_message.emit(f"----- Done {display_name} -----\n")

            os.unlink(temp_cfg.name)
            self.analysis_finished.emit(str(Path(output_dir)))
            self.update_progress.emit(100, "All video analysis completed!")

        except Exception as e:
            self.analysis_error.emit(str(e))

    def _run_one_video(self, vf: Path, model: str, cfg: str, display: bool, output_dir: Path):
        cmd = [
            'python', 'main.py',
            '--video', str(vf),
            '--model', model,
            '--config', cfg,
            '--output', str(output_dir)
        ]
        if display:
            cmd.append('--display')

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        prog_re = re.compile(r'(\d+)%')
        for line in iter(proc.stdout.readline, ''):
            if self.cancel_flag:
                proc.terminate()
                return
            line = line.rstrip()
            self.log_message.emit(line)
            m = prog_re.search(line)
            if m:
                self.update_progress.emit(int(m.group(1)), f"Progress: {m.group(1)}%")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Video analysis failed with return code: {rc}")

    def cancel(self):
        self.cancel_flag = True


# ------------------------- Training Worker Thread -------------------------
class TrainingWorker(QThread):
    log_message = pyqtSignal(str)
    finished = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, cmd_list, workdir=None, parent=None):
        super().__init__(parent)
        self.cmd_list = cmd_list
        self.workdir = workdir
        self._cancel = False
        self._proc = None

    def run(self):
        try:
            self.log_message.emit("Start training...")
            self.log_message.emit("Command: " + " ".join(self.cmd_list))

            self._proc = subprocess.Popen(
                self.cmd_list,
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

            for line in iter(self._proc.stdout.readline, ''):
                if self._cancel:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                    self.log_message.emit("Training canceled.")
                    self.finished.emit(-1, "Training canceled.")
                    return
                self.log_message.emit(line.rstrip())

            rc = self._proc.wait()
            if rc == 0:
                self.finished.emit(0, "Training finished.")
            else:
                self.finished.emit(rc, f"Training ended with return code: {rc}")
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancel = True
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass


# ------------------------- Main Window -------------------------
class PandaAnalysisSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panda Behavior Analysis System")
        self.setGeometry(100, 100, 1400, 900)
        self.result_df = None
        self.config = self.load_default_config()
        self.video_items = []
        self.train_worker = None
        self.init_ui()

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        lay = QVBoxLayout(main)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        self.setup_analysis_tab()
        self.setup_vis_tab()
        self.setup_training_tab()

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.status_label)

        footer = QLabel("© 2026 Panda Behavior Analysis System")
        footer.setAlignment(Qt.AlignCenter)
        lay.addWidget(footer)

    # ---------------- Analysis tab ----------------
    def setup_analysis_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Video Analysis")
        lay = QVBoxLayout(tab)

        title = QLabel("Panda Video Behavior Analysis")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        grp = QGroupBox("Settings")
        form = QFormLayout(grp)
        form.setLabelAlignment(Qt.AlignRight)

        video_list_grp = QGroupBox("Video Items")
        video_list_lay = QVBoxLayout(video_list_grp)

        btn_lay = QHBoxLayout()
        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self.add_video_folder)
        self.add_file_btn = QPushButton("Add File")
        self.add_file_btn.clicked.connect(self.add_video_file)
        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.clicked.connect(self.clear_video_list)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected_item)

        btn_lay.addWidget(self.add_folder_btn)
        btn_lay.addWidget(self.add_file_btn)
        btn_lay.addWidget(self.clear_btn)
        btn_lay.addWidget(self.remove_btn)

        self.video_tree = QTreeWidget()
        self.video_tree.setHeaderLabels(["Type", "Name", "Path"])
        self.video_tree.setColumnWidth(0, 90)
        self.video_tree.setColumnWidth(1, 240)
        self.video_tree.header().setSectionResizeMode(2, QHeaderView.Stretch)

        video_list_lay.addLayout(btn_lay)
        video_list_lay.addWidget(self.video_tree)
        form.addRow("Video Items:", video_list_grp)

        self.model_input = QLineEdit()
        btn2 = QPushButton("Browse...")
        btn2.clicked.connect(self.select_model)
        h2 = QHBoxLayout()
        h2.addWidget(self.model_input)
        h2.addWidget(btn2)
        form.addRow("Model:", h2)

        self.frame_skip_input = QLineEdit(str(self.config['frame_skip']))
        self.frame_skip_input.setValidator(QIntValidator(1, 10))
        form.addRow("Frame Skip:", self.frame_skip_input)

        self.show_display_cb = QCheckBox("Show Preview (Debug)")
        form.addRow("Display:", self.show_display_cb)

        self.output_dir_input = QLineEdit(self.config['output_dir'])
        btn4 = QPushButton("Browse...")
        btn4.clicked.connect(self.select_output_dir)
        h4 = QHBoxLayout()
        h4.addWidget(self.output_dir_input)
        h4.addWidget(btn4)
        form.addRow("Output Directory:", h4)

        lay.addWidget(grp)

        pg_grp = QGroupBox("Progress")
        pg_lay = QVBoxLayout(pg_grp)

        self.current_video_label = QLabel("Current: None")
        pg_lay.addWidget(self.current_video_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        pg_lay.addWidget(QLabel("Overall:"))
        pg_lay.addWidget(self.progress_bar)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        pg_lay.addWidget(QLabel("Log:"))
        pg_lay.addWidget(self.log_output)
        lay.addWidget(pg_grp)

        btn_lay2 = QHBoxLayout()
        self.analyze_btn = QPushButton("Start")
        self.analyze_btn.setStyleSheet("background:#27ae60;color:white;padding:10px;font-weight:bold;")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("background:#e74c3c;color:white;padding:10px;")
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        btn_lay2.addWidget(self.analyze_btn)
        btn_lay2.addWidget(self.cancel_btn)
        lay.addLayout(btn_lay2)

    # ---------------- Visualization tab ----------------
    def setup_vis_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Trajectory Visualization")
        lay = QVBoxLayout(tab)

        title = QLabel("Panda Trajectory Visualization")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        grp = QGroupBox("Data")
        form = QFormLayout(grp)

        self.load_btn = QPushButton("Load Trajectory Excel")
        self.load_btn.clicked.connect(self.load_file)
        self.load_btn.setStyleSheet("""
            QPushButton{background:#3498db;color:white;border:none;border-radius:5px;padding:6px 12px;font-weight:bold;min-width:120px;}
            QPushButton:hover{background:#2980b9;}
            QPushButton:pressed{background:#21618c;}
        """)
        form.addRow("Source:", self.load_btn)

        self.start_time_edit = QDateTimeEdit(QDateTime.currentDateTime())
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_time_edit.setCalendarPopup(True)
        form.addRow("Start Time:", self.start_time_edit)

        self.end_time_edit = QDateTimeEdit(QDateTime.currentDateTime().addSecs(3600))
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.end_time_edit.setCalendarPopup(True)
        form.addRow("End Time:", self.end_time_edit)

        self.background_input = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self.select_file(self.background_input, "Image (*.png *.jpg *.jpeg)"))
        h = QHBoxLayout()
        h.addWidget(self.background_input)
        h.addWidget(btn)
        form.addRow("Background Image:", h)

        self.generate_btn = QPushButton("Generate Plots")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.generate_visualizations)
        self.generate_btn.setStyleSheet("""
            QPushButton{background:#3498db;color:white;border:none;border-radius:5px;padding:6px 12px;font-weight:bold;min-width:120px;}
            QPushButton:hover{background:#2980b9;}
            QPushButton:pressed{background:#21618c;}
        """)
        form.addRow("", self.generate_btn)
        lay.addWidget(grp)

        splitter = QSplitter(Qt.Horizontal)
        self.heatmap_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.trajectory_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        splitter.addWidget(self._frame("Heatmap", self.heatmap_canvas))
        splitter.addWidget(self._frame("Trajectory", self.trajectory_canvas))
        lay.addWidget(splitter, 1)

        self.vis_status_label = QLabel("Please load a trajectory file.")
        self.vis_status_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.vis_status_label)

    # ---------------- Training tab ----------------
    def setup_training_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Model Training")
        lay = QVBoxLayout(tab)

        title = QLabel("Model Training (YOLO Dataset)")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        # 1) Optional: extract frames
        extract_grp = QGroupBox("1) Optional: Extract Frames from Video")
        extract_form = QFormLayout(extract_grp)

        self.train_video_input = QLineEdit()
        btn_video = QPushButton("Browse...")
        btn_video.clicked.connect(self._select_train_video)
        h1 = QHBoxLayout()
        h1.addWidget(self.train_video_input)
        h1.addWidget(btn_video)
        extract_form.addRow("Video File:", h1)

        self.frames_outdir_input = QLineEdit()
        btn_outdir = QPushButton("Browse...")
        btn_outdir.clicked.connect(self._select_frames_output_dir)
        h2 = QHBoxLayout()
        h2.addWidget(self.frames_outdir_input)
        h2.addWidget(btn_outdir)
        extract_form.addRow("Output Folder:", h2)

        self.frames_count_input = QLineEdit("300")
        self.frames_count_input.setValidator(QIntValidator(1, 1000000))
        extract_form.addRow("Number of Frames:", self.frames_count_input)

        self.frames_prefix_input = QLineEdit("frame")
        extract_form.addRow("Filename Prefix:", self.frames_prefix_input)

        btn_extract = QPushButton("Extract")
        btn_extract.clicked.connect(self._extract_frames_from_video)
        btn_extract.setStyleSheet("background:#8e44ad;color:white;padding:8px;font-weight:bold;")
        extract_form.addRow("", btn_extract)

        lay.addWidget(extract_grp)

        # 2) makesense.ai link
        label_grp = QGroupBox("2) Annotation (MakeSense.ai)")
        label_lay = QVBoxLayout(label_grp)
        label_info = QLabel("Click to open MakeSense.ai in your browser for labeling, then export a YOLO-format dataset.")
        label_info.setWordWrap(True)
        label_lay.addWidget(label_info)

        btn_open = QPushButton("Open MakeSense.ai")
        btn_open.clicked.connect(self._open_makesense)
        btn_open.setStyleSheet("background:#2980b9;color:white;padding:8px;font-weight:bold;")
        label_lay.addWidget(btn_open)
        lay.addWidget(label_grp)

        # 3) data.yaml + training
        train_grp = QGroupBox("3) Generate data.yaml and Train")
        train_form = QFormLayout(train_grp)

        self.ds_root_input = QLineEdit()
        btn_ds_root = QPushButton("Select Root")
        btn_ds_root.clicked.connect(self._pick_dataset_root)
        row_root = QHBoxLayout()
        row_root.addWidget(self.ds_root_input)
        row_root.addWidget(btn_ds_root)
        train_form.addRow("Dataset Root:", row_root)

        self.class_names_input = QLineEdit()
        self.class_names_input.setPlaceholderText("Optional: comma-separated names, e.g., panda,cat; leave empty to auto-generate class0,class1...")
        train_form.addRow("Class Names (names):", self.class_names_input)

        btn_gen_yaml = QPushButton("Generate data.yaml")
        btn_gen_yaml.clicked.connect(self._generate_data_yaml)
        train_form.addRow("", btn_gen_yaml)

        self.data_yaml_input = QLineEdit()
        btn_data = QPushButton("Browse...")
        btn_data.clicked.connect(lambda: self.select_file(self.data_yaml_input, "YAML (*.yaml *.yml)"))
        h3 = QHBoxLayout()
        h3.addWidget(self.data_yaml_input)
        h3.addWidget(btn_data)
        train_form.addRow("data.yaml:", h3)

        self.base_model_input = QLineEdit()
        btn_base = QPushButton("Browse...")
        btn_base.clicked.connect(lambda: self.select_file(self.base_model_input, "Model (*.pt)"))
        h4 = QHBoxLayout()
        h4.addWidget(self.base_model_input)
        h4.addWidget(btn_base)
        train_form.addRow("Base Model (.pt):", h4)

        self.train_project_out_input = QLineEdit(str(Path("runs") / "train"))
        btn_proj = QPushButton("Browse...")
        btn_proj.clicked.connect(self._select_train_project_dir)
        h5 = QHBoxLayout()
        h5.addWidget(self.train_project_out_input)
        h5.addWidget(btn_proj)
        train_form.addRow("Output Directory:", h5)

        self.epochs_input = QLineEdit("100")
        self.epochs_input.setValidator(QIntValidator(1, 100000))
        train_form.addRow("Epochs:", self.epochs_input)

        self.imgsz_input = QLineEdit("640")
        self.imgsz_input.setValidator(QIntValidator(32, 4096))
        train_form.addRow("Image Size (imgsz):", self.imgsz_input)

        self.batch_input = QLineEdit("16")
        self.batch_input.setValidator(QIntValidator(1, 1024))
        train_form.addRow("Batch Size:", self.batch_input)

        self.device_input = QLineEdit("gpu")
        train_form.addRow("Device (gpu / gpu:0 / cpu):", self.device_input)

        btn_train_row = QHBoxLayout()
        self.btn_start_train = QPushButton("Start Training")
        self.btn_start_train.clicked.connect(self._start_model_training)
        self.btn_start_train.setStyleSheet("background:#27ae60;color:white;padding:10px;font-weight:bold;")

        self.btn_cancel_train = QPushButton("Cancel Training")
        self.btn_cancel_train.clicked.connect(self._cancel_model_training)
        self.btn_cancel_train.setEnabled(False)
        self.btn_cancel_train.setStyleSheet("background:#c0392b;color:white;padding:10px;font-weight:bold;")

        btn_train_row.addWidget(self.btn_start_train)
        btn_train_row.addWidget(self.btn_cancel_train)
        train_form.addRow("", btn_train_row)

        lay.addWidget(train_grp)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        lay.addWidget(QLabel("Training Log:"))
        lay.addWidget(self.train_log, 1)

        self.train_status_label = QLabel("Ready")
        self.train_status_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.train_status_label)

    # ---------------- Training helpers ----------------
    def _select_train_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if f:
            self.train_video_input.setText(f)

    def _select_frames_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self.frames_outdir_input.setText(d)

    def _select_train_project_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Training Output Folder")
        if d:
            self.train_project_out_input.setText(d)

    def _open_makesense(self):
        QDesktopServices.openUrl(QUrl("https://www.makesense.ai/"))

    def _append_train_log(self, msg: str):
        self.train_log.append(msg)
        self.train_log.moveCursor(QTextCursor.End)

    def _extract_frames_from_video(self):
        video_path = self.train_video_input.text().strip()
        out_dir = self.frames_outdir_input.text().strip()
        n_str = self.frames_count_input.text().strip()
        prefix = self.frames_prefix_input.text().strip() or "frame"

        if not video_path or not Path(video_path).exists():
            QMessageBox.warning(self, "Error", "Please select a valid video file.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Error", "Please select an output folder.")
            return
        if not n_str:
            QMessageBox.warning(self, "Error", "Please set the number of frames.")
            return

        try:
            n_frames = int(n_str)
            if n_frames <= 0:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid frame count.")
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open the video.")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total // n_frames) if total > 0 else 1

        self._append_train_log(f"Extracting frames: total={total}, target={n_frames}, step={step}")
        saved = 0
        idx = 0
        frame_id = 0

        while cap.isOpened() and saved < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame_id += 1
                out_file = out_path / f"{prefix}_{frame_id:06d}.jpg"
                cv2.imwrite(str(out_file), frame)
                saved += 1
            idx += 1

        cap.release()
        self._append_train_log(f"Done. Saved {saved} images to: {out_path}")
        QMessageBox.information(self, "Done", f"Saved {saved} images to:\n{out_path}")

    def _normalize_ultralytics_device(self, device_text: str) -> str:
        s = (device_text or "").strip().lower()
        if not s:
            return "cpu"
        if s == "cpu":
            return "cpu"
        if s in ("gpu", "cuda"):
            return "0"
        if s.startswith("gpu:") or s.startswith("cuda:"):
            tail = s.split(":", 1)[1].strip()
            return tail if tail else "0"
        return s.replace(" ", "")

    def _pick_dataset_root(self):
        QMessageBox.information(
            self,
            "Hint",
            "Please place train/val/test under the same root directory and keep the folder structure, for example:\n"
            "train/images and train/labels\n"
            "val/images and val/labels\n"
            "test/images and test/labels (optional)"
        )
        d = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if d:
            self.ds_root_input.setText(d)

    def _infer_nc_from_labels(self, labels_dir: Path) -> int:
        max_cls = -1
        for p in labels_dir.rglob("*.txt"):
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    if cls_id > max_cls:
                        max_cls = cls_id
            except Exception:
                continue
        return max_cls + 1

    def _generate_data_yaml(self):
        root_str = self.ds_root_input.text().strip()
        if not root_str:
            QMessageBox.warning(self, "Error", "Please select the dataset root directory first.")
            return

        root = Path(root_str)
        if not root.exists():
            QMessageBox.warning(self, "Error", "The dataset root directory does not exist.")
            return

        # Supported layouts:
        # A) root/images/train and root/labels/train
        # B) root/train/images and root/train/labels (recommended)
        caseB_train = root / "train" / "images"
        caseB_val = root / "val" / "images"
        caseB_test = root / "test" / "images"

        caseA_train = root / "images" / "train"
        caseA_val = root / "images" / "val"
        caseA_test = root / "images" / "test"
        caseA_labels = root / "labels"

        train_spec = None
        val_spec = None
        test_spec = None
        labels_dir_for_nc = None

        if caseB_train.exists() and caseB_val.exists():
            train_spec = str(caseB_train.as_posix())
            val_spec = str(caseB_val.as_posix())
            if caseB_test.exists():
                test_spec = str(caseB_test.as_posix())
            labels_dir_for_nc = root / "train" / "labels"
            if not labels_dir_for_nc.exists():
                QMessageBox.warning(self, "Error", "Missing train/labels. Please check the dataset structure.")
                return
        elif caseA_train.exists() and caseA_val.exists() and caseA_labels.exists():
            train_spec = str(caseA_train.as_posix())
            val_spec = str(caseA_val.as_posix())
            if caseA_test.exists():
                test_spec = str(caseA_test.as_posix())
            labels_dir_for_nc = caseA_labels
        else:
            QMessageBox.warning(
                self, "Error",
                "Unrecognized YOLO dataset structure.\n\n"
                "Supported structures:\n"
                "1) root/images/train and root/labels/train\n"
                "2) root/train/images and root/train/labels (recommended)"
            )
            return

        nc = self._infer_nc_from_labels(labels_dir_for_nc)
        if nc <= 0:
            QMessageBox.warning(self, "Error", "No valid class IDs found in labels.")
            return

        names_text = self.class_names_input.text().strip()
        if names_text:
            names = [x.strip() for x in names_text.split(",") if x.strip()]
            if len(names) != nc:
                QMessageBox.warning(self, "Error", f"names count={len(names)} but inferred nc={nc}. Please match them or leave names empty.")
                return
        else:
            names = [f"class{i}" for i in range(nc)]

        data = {
            "path": str(root.as_posix()),
            "train": train_spec,
            "val": val_spec,
            "nc": int(nc),
            "names": names
        }
        if test_spec is not None:
            data["test"] = test_spec

        out_yaml = root / "data.yaml"
        out_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

        self.data_yaml_input.setText(str(out_yaml))
        self._append_train_log(f"Generated: {out_yaml}")
        QMessageBox.information(self, "Done", f"Generated data.yaml:\n{out_yaml}")

    def _start_model_training(self):
        if self.train_worker is not None and self.train_worker.isRunning():
            QMessageBox.warning(self, "Hint", "Training is already running.")
            return

        data_yaml = self.data_yaml_input.text().strip()
        base_model = self.base_model_input.text().strip()
        project_dir = self.train_project_out_input.text().strip()
        epochs = self.epochs_input.text().strip() or "100"
        imgsz = self.imgsz_input.text().strip() or "640"
        batch = self.batch_input.text().strip() or "16"

        raw_device = self.device_input.text().strip() or "cpu"
        device = self._normalize_ultralytics_device(raw_device)

        if not data_yaml or not Path(data_yaml).exists():
            QMessageBox.warning(self, "Error", "Please select a valid data.yaml.")
            return
        if not base_model or not Path(base_model).exists():
            QMessageBox.warning(self, "Error", "Please select a valid base model (.pt).")
            return
        if not project_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return

        # If GPU requested but CUDA is unavailable, fall back to CPU
        if device != "cpu":
            try:
                if (not torch.cuda.is_available()) or (torch.cuda.device_count() == 0):
                    self._append_train_log("No CUDA device detected by torch. Falling back to CPU training.")
                    QMessageBox.warning(
                        self, "GPU Unavailable",
                        "You requested GPU training, but CUDA was not detected.\n"
                        "Falling back to CPU.\n\n"
                        "To use GPU, install a CUDA-enabled PyTorch build and ensure GPU drivers are installed."
                    )
                    device = "cpu"
            except Exception as e:
                self._append_train_log(f"GPU check failed: {e}. Falling back to CPU.")
                device = "cpu"

        cmd = [
            "yolo", "detect", "train",
            f"data={data_yaml}",
            f"model={base_model}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"device={device}",
            f"project={project_dir}",
        ]

        self.train_log.clear()
        self._append_train_log("Preparing to start training...")
        self.train_status_label.setText("Training...")
        self.btn_start_train.setEnabled(False)
        self.btn_cancel_train.setEnabled(True)

        self.train_worker = TrainingWorker(cmd_list=cmd, workdir=None)
        self.train_worker.log_message.connect(self._append_train_log)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.finished.connect(self._on_train_finished)
        self.train_worker.start()

    def _cancel_model_training(self):
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.cancel()
            self.btn_cancel_train.setEnabled(False)
            self._append_train_log("Cancel requested...")

    def _on_train_error(self, err: str):
        self._append_train_log("Error: " + err)
        self.train_status_label.setText("Error")
        self.btn_start_train.setEnabled(True)
        self.btn_cancel_train.setEnabled(False)
        QMessageBox.critical(self, "Training Error", err)

    def _on_train_finished(self, rc: int, msg: str):
        self._append_train_log(msg)
        self.train_status_label.setText("Ready" if rc == 0 else "Finished")
        self.btn_start_train.setEnabled(True)
        self.btn_cancel_train.setEnabled(False)
        if rc == 0:
            QMessageBox.information(self, "Training", msg)
        else:
            QMessageBox.warning(self, "Training", msg)

    # ---------------- Common helpers ----------------
    def _frame(self, title, canvas):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(frame)
        lab = QLabel(title)
        lab.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        lab.setAlignment(Qt.AlignCenter)
        lay.addWidget(lab)
        lay.addWidget(canvas)
        return frame

    def load_default_config(self):
        return {'output_dir': 'results', 'frame_skip': 2, 'show_display': False}

    def add_video_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder:
            folder_name = Path(folder).name
            for item in self.video_items:
                if isinstance(item, dict) and item.get('path') == folder:
                    QMessageBox.warning(self, "Duplicate", f"Folder '{folder_name}' already exists.")
                    return

            self.video_items.append({'type': 'folder', 'name': folder_name, 'path': folder})

            folder_item = QTreeWidgetItem(self.video_tree)
            folder_item.setText(0, "Folder")
            folder_item.setText(1, folder_name)
            folder_item.setText(2, folder)

            mp4_count = 0
            for mp4_file in Path(folder).rglob("*.mp4"):
                child_item = QTreeWidgetItem(folder_item)
                child_item.setText(0, "Video")
                child_item.setText(1, mp4_file.name)
                child_item.setText(2, str(mp4_file))
                mp4_count += 1

            if mp4_count == 0:
                folder_item.setText(1, f"{folder_name} (no MP4 found)")

            self.video_tree.expandAll()

    def add_video_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", "Video (*.mp4 *.avi *.mov)")
        for file in files:
            file_path = Path(file)
            if str(file_path) in self.video_items:
                continue
            self.video_items.append(str(file_path))

            file_item = QTreeWidgetItem(self.video_tree)
            file_item.setText(0, "Video")
            file_item.setText(1, file_path.name)
            file_item.setText(2, str(file_path))

    def clear_video_list(self):
        self.video_items.clear()
        self.video_tree.clear()

    def remove_selected_item(self):
        selected_items = self.video_tree.selectedItems()
        for item in selected_items:
            parent = item.parent()
            if parent:
                parent.removeChild(item)
                if parent.childCount() == 0:
                    root = self.video_tree.invisibleRootItem()
                    root.removeChild(parent)
                    folder_path = parent.text(2)
                    self.video_items = [it for it in self.video_items
                                        if not (isinstance(it, dict) and it.get('path') == folder_path)]
            else:
                index = self.video_tree.indexOfTopLevelItem(item)
                self.video_tree.takeTopLevelItem(index)
                if item.text(0) == "Folder":
                    folder_path = item.text(2)
                    self.video_items = [it for it in self.video_items
                                        if not (isinstance(it, dict) and it.get('path') == folder_path)]
                else:
                    file_path = item.text(2)
                    self.video_items = [it for it in self.video_items if it != file_path]

    def select_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model (*.pt)")
        if f:
            self.model_input.setText(f)

    def select_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.output_dir_input.setText(d)

    def select_file(self, line_edit, filter_str):
        f, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if f:
            line_edit.setText(f)

    # ---------------- Analysis actions ----------------
    def start_analysis(self):
        if not self.video_items:
            QMessageBox.warning(self, "Warning", "Please add video files or folders first.")
            return
        if not self.model_input.text().strip():
            QMessageBox.warning(self, "Warning", "Please select a model file.")
            return
        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory.")
            return

        cfg = {
            'video_items': self.video_items,
            'model_path': self.model_input.text().strip(),
            'output_dir': output_dir,
            'frame_skip': self.frame_skip_input.text().strip(),
            'show_display': self.show_display_cb.isChecked()
        }

        self.analyze_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.current_video_label.setText("Current: Preparing...")

        self.worker = AnalysisWorker(cfg)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.video_progress.connect(self.update_video_progress)
        self.worker.log_message.connect(self.log_output.append)
        self.worker.analysis_finished.connect(self.analysis_finished)
        self.worker.analysis_error.connect(self.analysis_error)
        self.worker.finished.connect(self.analysis_completed)
        self.worker.start()

    def cancel_analysis(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, "Confirm", "Cancel analysis?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.cancel_btn.setEnabled(False)
                self.log_output.append("Analysis canceled.")

    def update_progress(self, v, msg):
        self.progress_bar.setValue(v)
        self.status_label.setText(msg)

    def update_video_progress(self, video_name, progress):
        self.current_video_label.setText(f"Current: {video_name} ({progress}%)")

    def analysis_finished(self, p):
        self.log_output.append(f"Done. Results saved to: {p}")
        QMessageBox.information(self, "Done", f"All videos processed.\nResults saved to: {p}")
        self.current_video_label.setText("Current: Completed")

    def analysis_error(self, e):
        self.log_output.append(f"Error: {e}")
        QMessageBox.critical(self, "Error", str(e))

    def analysis_completed(self):
        self.analyze_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Ready")

    # ---------------- Visualization actions ----------------
    def load_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Trajectory Excel", "", "Excel (*.xlsx *.xls)")
        if f:
            try:
                self.result_df = pd.read_excel(f)
                self.result_df['datetime'] = pd.to_datetime(self.result_df['datetime'], format='%Y%m%d_%H_%M_%S')
                self.start_time_edit.setDateTime(self.result_df['datetime'].min())
                self.end_time_edit.setDateTime(self.result_df['datetime'].max())
                self.vis_status_label.setText(f"Loaded: {f} | Rows: {len(self.result_df)}")
                self.generate_btn.setEnabled(True)
            except Exception as e:
                self.vis_status_label.setText(f"Load failed: {e}")

    def generate_visualizations(self):
        if self.result_df is None:
            return
        s = self.start_time_edit.dateTime().toPyDateTime()
        e = self.end_time_edit.dateTime().toPyDateTime()
        df = self.result_df[(self.result_df['datetime'] >= s) & (self.result_df['datetime'] <= e)]
        if df.empty:
            self.vis_status_label.setText("No data in the selected time range.")
            return
        self._plot_heatmap(df)
        self._plot_trajectory(df)
        self.vis_status_label.setText(f"Plots generated (rows: {len(df)})")

    def _plot_heatmap(self, df):
        self.heatmap_canvas.figure.clear()
        ax = self.heatmap_canvas.figure.add_subplot(111)
        w, h = 1920, 1080
        if 'frame_width' in df.columns:
            w = int(df['frame_width'].iloc[0])
            h = int(df['frame_height'].iloc[0])
        heat = np.zeros((h, w))
        for _, r in df.iterrows():
            if pd.notna(r.get('x')) and pd.notna(r.get('y')):
                x, y = int(r['x']), int(r['y'])
                if 0 <= x < w and 0 <= y < h:
                    dtv = r.get('dwell_time', 1)
                    for dx in range(-5, 6):
                        for dy in range(-5, 6):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                heat[ny, nx] += np.exp(-(dx ** 2 + dy ** 2) / 10) * max(1, dtv)
        im = ax.imshow(heat, cmap='hot', interpolation='gaussian', alpha=0.8)
        self.heatmap_canvas.figure.colorbar(im, ax=ax, label='Activity Intensity')
        ax.set_title(f"Heatmap (points: {len(df)})")
        ax.grid(False)
        self.heatmap_canvas.draw()

    def _plot_trajectory(self, df):
        self.trajectory_canvas.figure.clear()
        ax = self.trajectory_canvas.figure.add_subplot(111)
        w, h = 1920, 1080
        if 'frame_width' in df.columns:
            w = int(df['frame_width'].iloc[0])
            h = int(df['frame_height'].iloc[0])
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')

        prev = None
        for _, r in df.iterrows():
            if pd.notna(r.get('x')) and pd.notna(r.get('y')):
                x, y = int(r['x']), int(r['y'])
                state = r.get('state')
                c, m, s = (('#e74c3c', 'o', 6) if state == 'moving' else ('#3498db', 's', 6))
                ax.plot(x, y, marker=m, markersize=s, color=c, alpha=0.6, markeredgecolor='white')
                if prev is not None:
                    ax.plot([prev[0], x], [prev[1], y], '-', color='#2ecc71', alpha=0.5, lw=1.5)
                prev = (x, y)

        ax.legend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db', markersize=8),
            Line2D([0], [0], color='#2ecc71', lw=2)
        ], ['Moving', 'Stationary', 'Trajectory'], loc='upper right')

        ax.set_title("Trajectory")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.1)
        self.trajectory_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    w = PandaAnalysisSystem()
    w.show()
    sys.exit(app.exec_())
