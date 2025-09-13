# Brain Tumor Detection with YOLOv8

## Project Overview
This project uses the YOLOv8 object detection model to detect brain tumors in medical images. The workflow includes dataset preparation, model training, evaluation, and inference on new images. All code and results are provided in the Jupyter notebook `YOLOv8_BrainTumor_Training.ipynb`.

## Directory Structure
```
Cancer_Detection/
│
├── YOLOv8_BrainTumor_Training.ipynb   # Main notebook for data prep, training, and inference
├── brain-tumor-sample.jpg             # Example image for inference
├── GlioblastomaBranScans-700x467.jpg  # Example image for inference
├── ijms-23-01187-g001.png             # Example image for inference
├── Screenshot 2025-09-13 234516.png   # Example image for inference
├── yolov8n.pt                         # Pretrained YOLOv8 weights
│
├── BrainTumor/
│   └── BrainTumorYolov8/
│       ├── data.yaml                  # Dataset config for YOLOv8
│       ├── train/
│       │   ├── images/                # Training images
│       │   └── labels/                # YOLO-format labels for training
│       ├── valid/
│       │   ├── images/                # Validation images
│       │   └── labels/                # YOLO-format labels for validation
│       └── test/
│           ├── images/                # Test images
│           └── labels/                # YOLO-format labels for test
│
├── models/
│   └── yolov8_brain_tumor.pt         # Trained YOLOv8 model weights
│
├── runs/
│   ├── cancer_detection/              # Inference results for brain-tumor-sample.jpg
│   ├── cancer_detection2/             # Inference results for GlioblastomaBranScans-700x467.jpg
│   └── cancer_detection3/             # Inference results for ijms-23-01187-g001.png
│
└── yolov8_brain_tumor/
    └── exp1/
        ├── weights/
        │   ├── best.pt                # Best model checkpoint
        │   └── last.pt                # Last model checkpoint
        ├── *.png, *.jpg, *.csv        # Training logs, plots, and results
        └── ...
```

## How to Use
1. **Dataset Preparation:**
   - The dataset is organized in YOLO format under `BrainTumor/BrainTumorYolov8/` with `train`, `valid`, and `test` splits.
   - The `data.yaml` file defines the dataset structure and class names.

2. **Model Training:**
   - Run the notebook `YOLOv8_BrainTumor_Training.ipynb` to train the YOLOv8 model on the custom dataset.
   - Training outputs (weights, logs, plots) are saved in `yolov8_brain_tumor/exp1/`.
   - The best model is also saved as `models/yolov8_brain_tumor.pt`.

3. **Inference:**
   - Use the notebook to run predictions on new images.
   - Results are saved in the `runs/cancer_detection*` folders.

4. **Visualization:**
   - Training and validation results, confusion matrices, and sample predictions are available as images in `yolov8_brain_tumor/exp1/`.

## Requirements
- Python 3.8+
- ultralytics
- opencv-python
- matplotlib
- kagglehub (for dataset download)


## License & Copyright

Copyright © Jaikishan Nishad, 2025.

This project and its contents were created by Jaikishan Nishad. All rights reserved.

You may use, share, or modify this project for educational and research purposes, provided proper attribution is given. For commercial use or other inquiries, please contact the author directly.

**Author:** Jaikishan Nishad  
**GitHub:** [JaikishanNishad](https://github.com/Jaikishan1228)  
**Contact:** jaikishannishad33@gmail.com  
**Date:** 14th September 2025
