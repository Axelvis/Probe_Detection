# Probe Detection Using Faster R-CNN

This repository contains the code, model weights, and instructions for a deep learning-based system to detect probes in images using Faster R-CNN with a ResNet-50 backbone.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)

---

## Overview

This project aims to automate the detection of probes in industrial or medical images, facilitating real-time inspection tasks. The system is implemented using **Faster R-CNN** pre-trained on the COCO dataset, fine-tuned for the specific probe detection task.

---

## Features

- **State-of-the-Art Model**: Faster R-CNN with ResNet-50 backbone.
- **Configurable Pipeline**: Train or test with minimal configuration.
- **Resource Assessment**: Evaluate feasibility for deployment on drones and IoT boards (e.g., NVIDIA Jetson family).
- **Modular Codebase**: Organized with reusable functions in `utils.py`.

---

## Requirements

- Python 3.8 or later
- PyTorch 1.12.0 or later
- torchvision 0.13.0 or later
- Other libraries: `numpy`, `matplotlib`, `tqdm`, `pillow`, `opencv-python`

The trained model weights can be downloaded [here](https://drive.google.com/file/d/1HaeDpM5oE94iA5TCPNm2lVG00HR5vhjl/view?usp=drive_link).

Install the required dependencies using:
```bash
pip install -r requirements.txt

