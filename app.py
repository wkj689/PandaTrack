from flask import Flask, request, jsonify
from pathlib import Path
import logging
import time
import base64
import numpy as np
import cv2
from werkzeug.utils import secure_filename

from main import EnhancedPandaMonitor

app = Flask(__name__)

# =========================
# 1) 基础配置
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("panda_api")

# 模型路径：改成你自己的 best.pt 真实路径
MODEL_PATH = r"E:\Panda_1\best.pt"

# 上传视频保存目录
UPLOAD_DIR = Path(r"E:\Panda_1\uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 允许上传的视频格式
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# 你的原始配置
CONFIG = {
    "general": {
        "output_dir": "./output",
        "frame_skip": 2,
        "show_heatmap": False,
        "save_trajectory_plot": True,
        "database_path": "./database.sqlite"
    },
    "detection": {
        "conf_threshold": 0.4
    },
    "motion": {
        "bbox_shift_threshold": 0.02,
        "bbox_size_change": 0.08
    },
    "grid_analysis": {
        "rows": 8,
        "cols": 8,
        "show_grid": True
    }
}

# =========================
# 2) 全局加载模型（只加载一次）
# =========================
logger.info("Loading YOLO model...")
monitor = EnhancedPandaMonitor(model_path=MODEL_PATH, cfg=CONFIG)
logger.info("Model loaded successfully.")


# =========================
# 3) 健康检查
# =========================
@app.route("/", methods=["GET"])
def health():
    return "PandaTrack"


# =========================
# 4) 旧接口：传本地视频路径分析
# =========================
@app.route("/start_video_analysis", methods=["POST"])
def start_video_analysis():
    data = request.get_json(silent=True) or {}
    video_path = data.get("video_path", "")

    if not video_path:
        return jsonify({
            "status": "fail",
            "error": "Missing video_path"
        }), 400

    p = Path(video_path)
    if not p.exists():
        return jsonify({
            "status": "fail",
            "error": f"Video not found: {video_path}"
        }), 400

    try:
        result = monitor.process_video(str(p), show=False)
        return jsonify({
            "status": "success",
            "result": result
        })
    except Exception as e:
        logger.exception("Error processing video by path")
        return jsonify({
            "status": "fail",
            "error": str(e)
        }), 500


# =========================
# 5) 上传视频分析接口
# =========================
@app.route("/analyze_video_upload", methods=["POST"])
def analyze_video_upload():
    """
    Postman:
    POST /analyze_video_upload
    Body -> form-data
      file: 选择视频文件
      frame_skip: 可选，整数
    """
    if "file" not in request.files:
        return jsonify({
            "status": "fail",
            "error": "No file field. Use form-data key='file'"
        }), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({
            "status": "fail",
            "error": "Empty filename"
        }), 400

    # 可选 frame_skip
    frame_skip = request.form.get("frame_skip")
    if frame_skip is not None and frame_skip != "":
        try:
            CONFIG["general"]["frame_skip"] = int(frame_skip)
        except Exception:
            return jsonify({
                "status": "fail",
                "error": "frame_skip must be int"
            }), 400

    filename = secure_filename(f.filename)
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        return jsonify({
            "status": "fail",
            "error": f"Unsupported file type: {ext}",
            "allowed": sorted(list(ALLOWED_EXT))
        }), 400

    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = UPLOAD_DIR / f"{Path(filename).stem}_{ts}{ext}"
    f.save(str(save_path))
    logger.info(f"Uploaded video saved: {save_path}")

    try:
        result = monitor.process_video(str(save_path), show=False)
        return jsonify({
            "status": "success",
            "uploaded_video": str(save_path),
            "result": result
        })
    except Exception as e:
        logger.exception("Error processing uploaded video")
        return jsonify({
            "status": "fail",
            "uploaded_video": str(save_path),
            "error": str(e)
        }), 500


# =========================
# 6) 新接口：单帧 Base64 图片检测
# =========================
@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    """
    输入:
    {
      "image": "Base64图片字符串"
    }

    输出:
    {
      "detected": true/false,
      "confidence": 0.95,
      "bbox": [x, y, w, h]
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        image_b64 = data.get("image", "")

        if not image_b64:
            return jsonify({
                "detected": False,
                "confidence": 0,
                "bbox": [],
                "error": "Missing image field"
            }), 400

        # 兼容 data:image/jpeg;base64,xxxx 这种格式
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]

        # Base64 解码
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "detected": False,
                "confidence": 0,
                "bbox": [],
                "error": "Invalid image data"
            }), 400

        # 调用 YOLO 检测
        results = monitor.model.predict(
            source=img,
            conf=CONFIG["detection"]["conf_threshold"],
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return jsonify({
                "detected": False,
                "confidence": 0,
                "bbox": []
            })

        # 优先取 panda 类；若无法判断类别则取最高置信度框
        valid_boxes = []
        for box in boxes:
            try:
                cls_id = int(box.cls[0])
            except Exception:
                cls_id = None

            if monitor.panda_cls_id is None:
                valid_boxes.append(box)
            elif cls_id == monitor.panda_cls_id:
                valid_boxes.append(box)

        if len(valid_boxes) == 0:
            return jsonify({
                "detected": False,
                "confidence": 0,
                "bbox": []
            })

        best_box = max(valid_boxes, key=lambda b: float(b.conf[0]))

        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        confidence = float(best_box.conf[0])

        return jsonify({
            "detected": True,
            "confidence": round(confidence, 4),
            "bbox": [
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ]
        })

    except Exception as e:
        logger.exception("Error in detect_frame")
        return jsonify({
            "detected": False,
            "confidence": 0,
            "bbox": [],
            "error": str(e)
        }), 500


if __name__ == "__main__":
    # 关闭 debug 自动重载，更适合接口服务
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)