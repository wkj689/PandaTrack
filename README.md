# PandaTrack: A Computer Vision-Based Quantitative Spatial Behavioral Monitoring System for Captive Giant Pandas

---

## Introduction

PandaTrack is an end-to-end computer vision framework for automated detection, tracking, and spatial behavior analysis of captive giant pandas in zoo environments.

The system integrates deep learning-based object detection with spatial statistical analysis, enabling transformation of raw surveillance videos into structured behavioral representations, including:

- Individual movement trajectories  
- Grid-based enclosure utilization patterns  
- Spatial hotspot distributions  
- Behavioral spatial heterogeneity analysis  

Compared with traditional manual observation, PandaTrack provides a quantitative, reproducible, and scalable framework for animal behavior analysis.

---


```md id="fix1"
## Installation

```bash
git clone https://github.com/yourusername/PandaTrack.git
cd PandaTrack
pip install -r requirements.txt


## Usage

The PandaTrack system provides an integrated workflow consisting of three core functional modules.

These modules support video-based behavior analysis, trajectory visualization, and model training.

Together, they form an end-to-end pipeline from raw video input to spatial behavioral analytics.

---

## 1. Video Analysis Module

In this module, users can import raw video data for automated behavioral analysis.

The system supports both single-video input and batch processing.

After loading the trained detection model, PandaTrack performs frame-by-frame inference to detect panda instances and extract spatial positions.

---

## 2. Trajectory and Spatial Visualization Module

This module enables secondary analysis of trajectory data by importing generated trajectory files (.xlsx).

It supports the following functions:

- Temporal filtering of trajectory data  
- Generation of individual movement trajectory maps  
- Grid-based spatial heatmap construction  
- Visualization of spatial density distributions  

By integrating temporal and spatial dimensions, the system enables quantitative assessment of habitat utilization patterns and behavioral space preferences.

---

## 3. Model Training Module

This module enables users to train customized detection models using their own datasets.

The workflow includes:

- Extracting frames from raw video sequences  
- Performing annotation using external tools (e.g., MakeSense.ai)  
- Generating YOLO-format dataset configuration files (data.yaml)  
- Selecting pretrained weights  
- Training model with real-time monitoring of performance metrics  

---

## Video Naming Format Instructions

To ensure correct parsing of video timestamps, all video files should follow the format below:

```text id="qk9n3c"
YYYYMMDD_startHour_endHour.mp4







