# project-4--decodelabs-internship
#  DecodeLabs Industrial Training Kit - Project 4

## Computer Vision Suite: OCR & Object Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Tesseract](https://img.shields.io/badge/Tesseract-5.x-red.svg)](https://github.com/tesseract-ocr/tesseract)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Validation Checkpoints](#validation-checkpoints)
- [Output Examples](#output-examples)
- [Troubleshooting](#troubleshooting)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

---

##  Overview

This repository contains **two complete computer vision pipelines** as part of the DecodeLabs Industrial Training Kit:

| Module | Function | Technology |
|--------|----------|------------|
| ** OCR Pipeline** | Text extraction from images | Tesseract + OpenCV |
| ** Object Detection** | Object localization & identification | OpenCV DNN + Contour Detection |

Both pipelines achieve **≥80% confidence accuracy** as per industrial standards.

---

##  Features

### Common Features
- End-to-end processing pipelines
-  80% confidence threshold enforcement
- Visual confirmation with bounding boxes
- Automatic benchmark reporting
- Modular, reusable architecture

### OCR Specific
- Multi-PSM mode support (3,6,7,11)
- Advanced preprocessing (deskewing, noise removal)
- Character whitelist filtering
- Confidence-based text filtering

### Object Detection Specific
- Non-Maximum Suppression (NMS)
- Multi-class detection support
- Confidence bar charts
- Bounding box visualization

---
output will be:
<img width="1365" height="711" alt="image" src="https://github.com/user-attachments/assets/b950315f-078c-4d1a-85b6-7136dfb03daf" />
<img width="1360" height="722" alt="image" src="https://github.com/user-attachments/assets/d3694e9c-822e-41e9-acf3-e28dd3143e94" />
<img width="1365" height="722" alt="image" src="https://github.com/user-attachments/assets/ac62cc54-20c7-48a9-8b58-09acd6008c1d" />
<img width="1364" height="722" alt="image" src="https://github.com/user-attachments/assets/330e6ddf-31d7-4da7-979d-ca82a89fa3c5" />

---
##  Installation:
### Prerequisites
```bash
Python >= 3.8
pip >= 20.0
pip install -r requirements.txt
###installation option :
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH:
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
##macos
brew install tesseract
##linux
sudo apt update && sudo apt install tesseract-ocr
##verify installation
python -c "import cv2, pytesseract, numpy, matplotlib; print('✅ All libraries installed')"
 Author:
DecodeLabs AI Engineering Team
Batch: 2026
Module: Industrial Training Kit - Project 4
