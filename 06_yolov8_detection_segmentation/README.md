# 06 - YOLOv8 Object Detection & Instance Segmentation + Roboflow Adapter

Unified Ultralytics YOLOv8n detection (`yolov8n.pt`) and instance segmentation (`yolov8n-seg.pt`) fine-tuning, FPS latency benchmarking, and Roboflow dataset adapter.

## Architecture & Methodology
- **Detection (`yolov8n.pt`)**: Anchor-free CSPDarknet/PANet detector evaluated on Box mAP@50, Box mAP@50-95, Precision, Recall, and steady-state FPS.
- **Instance Segmentation (`yolov8n-seg.pt`)**: Joint bounding-box + ProtoNet mask head evaluated on Box & Mask mAP@50, Mask mAP@50-95, Precision, Recall, and FPS.
- **Roboflow Drop-In Adapter (`src/roboflow_adapter.py`)**: Validates Roboflow-exported YOLOv8 `data.yaml`, parses/formats normalized bounding-box and polygon label files, and converts polygon masks to bounding boxes.
- **Windows & 4 GB VRAM Safe**: Runs with `workers=0`, `imgsz=320`, `batch=4`, and stores all weights/runs cleanly inside `artifacts/`.

## Dataset Provenance
- **Official Ultralytics COCO8**: `coco8.yaml` (detection) & `coco8-seg.yaml` (instance segmentation), automatically downloaded and cached under `data/`.

## Reproduction
```bash
# Run offline CPU unit tests (Roboflow adapter, metrics extraction, plotting)
pytest tests/

# Fine-tune and benchmark YOLOv8n vs YOLOv8n-seg on COCO8
python train.py --epochs 10 --batch-size 4 --imgsz 320 --seed 42
```

## Outputs
- `artifacts/yolov8n_coco8_best.pt` & `artifacts/yolov8n_seg_coco8_best.pt`
- `reports/metrics.json` - Box/Mask mAP, Precision/Recall, FPS/ms-per-frame, and VRAM telemetry.
- `reports/yolo_det_vs_seg.png` - 2x2 real COCO detection vs mask comparison + benchmark bar plots.
