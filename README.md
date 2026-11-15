# AI/ML-Based Autonomous Garbage Sorting Robot

**Simulation and training of an autonomous waste-sorting robot in a virtual environment**

A supervised research project at **SRM University AP** (Department of CSE, AI/ML). The system looks at waste on a moving conveyor, decides which of five categories each item belongs to, and routes it to the correct bin. Everything is validated in simulation: a 2D Material Recovery Facility (MRF) simulator and a 3D Unity factory scene with an inverse-kinematics robot arm.

The five categories follow the four-stream segregation mandate of **India's Solid Waste Management Rules 2026**, rather than the legacy categories used by most public waste datasets.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Waste Taxonomy](#waste-taxonomy)
3. [System Architecture](#system-architecture)
4. [Datasets](#datasets)
5. [Models and Results](#models-and-results)
6. [Confidence-Based Rejection](#confidence-based-rejection)
7. [Explainability (Grad-CAM)](#explainability-grad-cam)
8. [Simulations](#simulations)
9. [Repository Structure](#repository-structure)
10. [Getting Started](#getting-started)
11. [Training Notes](#training-notes)
12. [Current Status and Limitations](#current-status-and-limitations)
13. [Roadmap](#roadmap)
14. [Acknowledgements](#acknowledgements)

---

## Highlights

- **Dual-model perception.** A MobileNetV2 classifier for single-item accuracy and interpretability, plus a YOLOv8s detector for locating several items in a cluttered scene.
- **Regulation-grounded classes.** Five waste categories mapped to the SWM Rules 2026 streams.
- **Safety-first decisions.** Predictions below a confidence threshold go to a manual-review lane instead of being forced into a bin.
- **Two simulators.** A Python/OpenCV MRF simulator with live YOLO inference, and a Unity 6 HDRP factory scene with a code-driven robot arm.
- **Explainable.** Grad-CAM heatmaps show what the classifier is looking at, and exposed dataset problems along the way.
- **Interactive demo.** A Gradio app for uploading any waste photo and seeing detections, confidences, and bin assignments.

---

## Waste Taxonomy

| Class | Examples | Bin colour | SWM Rules 2026 stream |
|---|---|---|---|
| Wet/Organic | Food scraps, peels, biodegradable matter | Green | Wet waste |
| Recyclable/Plastic | Bottles, containers, packaging | Blue | Dry waste |
| Hazardous | Batteries, aerosols, chemical containers, e-waste | Red | Special care waste |
| Health/Medical | Medicine packaging, gloves, masks, syringes | Orange | Sanitary waste |
| Glass/Metal | Glass bottles, metal cans, tins | Purple | Dry waste |

---

## System Architecture

```
Waste item on conveyor
        │
        ▼
 AI camera zone (start of belt)
        │   frame
        ▼
 Perception: YOLOv8s detector  +  MobileNetV2 classifier
        │   class + confidence
        ▼
 Confidence check ──► below threshold ──► Manual-review lane
        │
        ▼ above threshold
 Decision controller (item ID + class travel with the item)
        │
        ▼
 Actuation: diverter gate (2D)  /  IK robot arm (Unity 3D)
        │
        ▼
 Correct colour-coded bin  ──►  Live analytics dashboard
```

Detection happens once, at the start of the belt. The class decision then travels with the item as metadata, and the sorting mechanism fires when the item reaches the right position. In the 2D simulator this is time-based gate actuation (travel time = gate distance / belt speed), which mirrors how pneumatic sorters work in real recycling plants.

---

## Datasets

### Classification (5,950 images)

| Class | Images | Source |
|---|---|---|
| Wet/Organic | 699 | Kaggle `sumn2u/garbage-classification-v2` (biological) |
| Recyclable/Plastic | 1,597 | Kaggle (plastic) |
| Hazardous | 756 | Kaggle (battery) |
| Health/Medical | 232 | Web-sourced images of packaged medical items |
| Glass/Metal | 2,666 | Kaggle (metal + glass) |

Split 80/10/10 (train/val/test), stratified by class. Training-time augmentation: horizontal flip, ±15° rotation, brightness/contrast jitter.

A note on Health/Medical: the publicly available medical-waste dataset I first tried contained graphic biomedical imagery, so I replaced it with non-graphic images of unused packaged items (blister packs, sealed syringes, masks, gloves) gathered with targeted search queries.

### Detection (multi-item scenes)

The detector is trained on the Roboflow Universe **YOLO Waste Detection** dataset (11,466 train / 1,092 val / 546 test images, 42 original classes). A remapping script collapses the 42 labels into the project's classes. That dataset has no medical-waste labels, so the real-scene detector covers four of the five classes (Wet/Organic, Recyclable/Plastic, Hazardous, Glass/Metal); Health/Medical is handled by the classifier.

---

## Models and Results

### MobileNetV2 classifier

ImageNet-pretrained MobileNetV2, final layer replaced with a 5-way head. Adam, lr 1e-3, StepLR, 20 epochs, batch 32, 224×224 input.

- Train accuracy 93.57%, validation accuracy 93.05%
- **Test accuracy: 91.26%** (595 images)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Wet/Organic | 0.908 | 0.908 | 0.908 | 261 |
| Recyclable/Plastic | 0.958 | 0.986 | 0.972 | 70 |
| Hazardous | 0.870 | 0.833 | 0.851 | 24 |
| Health/Medical | 0.868 | 0.857 | 0.863 | 154 |
| Glass/Metal | 0.977 | 0.988 | 0.983 | 86 |

The main error mode is confusion between Wet/Organic and Health/Medical (about 40 samples across the two directions). Hazardous has only 24 test images, so its numbers are the least statistically reliable.

### YOLOv8s detector

Trained for 50 epochs on a Kaggle T4 GPU.

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Wet/Organic | 0.749 | 0.716 | 0.783 | 0.574 |
| Recyclable/Plastic | 0.792 | 0.617 | 0.701 | 0.455 |
| Hazardous | 0.771 | 0.675 | 0.778 | 0.600 |
| Glass/Metal | 0.886 | 0.800 | 0.851 | 0.562 |
| **All** | **0.799** | **0.702** | **0.778** | **0.548** |

Inference is about 5.9 ms per image on a T4 at validation time.

---

## Confidence-Based Rejection

An item is only routed to a bin if the model is confident enough:

```
decision(x) = argmax_k p_k(x)    if max_k p_k(x) >= tau
            = MANUAL REVIEW      otherwise
```

In a virtual sorting run over the 595 held-out test images (MobileNetV2, tau = 0.70):

| Outcome | Count | Share |
|---|---|---|
| Correctly sorted | 506 | 85.0% |
| Wrong bin | 26 | 4.4% |
| Sent to manual review | 63 | 10.6% |
| **Accuracy on accepted items** | | **95.1%** |

The 2D simulator uses tau = 0.35 for live YOLO detections. The threshold is a tunable trade-off between accuracy and how much work is pushed to a human.

---

## Explainability (Grad-CAM)

Grad-CAM on the last convolutional block of MobileNetV2 produced a few findings worth keeping:

- **Mislabelled training data.** The highest-confidence "Wet/Organic" sample showed tin cans, which pointed to label noise in the source dataset.
- **Symbol dependence.** For Hazardous, attention concentrated on a printed hazard symbol rather than the object's material. That works on labelled items but would likely fail on an unmarked battery, which argues for extra sensors (for example, inductive or near-IR) in any physical build.
- **Source of class confusion.** The Health/Medical vs Wet/Organic confusion traces back to visually similar packaging and textures.

---

## Simulations

### 2D MRF simulator (`mrf_simulator.py`)

A Python/OpenCV industrial-HMI style view:

- Animated conveyor belt with adjustable speed
- Items drawn from real dataset images
- AI camera zone where YOLOv8s runs live inference
- Five position-tracked diverter gates over colour-coded bins, plus a reject lane
- Dashboard with totals, per-class counts, accuracy, rejection rate, bin-full and low-confidence alerts

Controls: `SPACE` pause/resume, `+` / `-` belt speed, `R` reset stats, `Q` quit.

### Unity 3D simulation (Unity 6, HDRP)

Built on the **UnityFactorySceneHDRP** asset (Unity Technologies Japan) for a photorealistic factory interior with conveyor lines and a robot arm.

| Script | Role |
|---|---|
| `WasteItem.cs` | Item type, colour coding, waypoint movement along the belt |
| `WasteSpawner.cs` | Spawns items at the belt start and triggers sorting at the pick point |
| `SortingController.cs` | Drives the arm's IK target through pick, lift, transport, drop, return |
| `BinSetup.cs` | Creates five colour-coded, labelled bins at runtime |
| `AIClassifier.cs` | Captures a camera frame and queries the inference server |
| `ArmIK.cs` | Asset-provided 2-joint inverse kinematics (law of cosines) |

Items follow a waypoint path along the conveyor. The arm moves an IK target through a fixed sequence and the joint angles follow from the IK solver.

### Bridge between Unity and the model

`inference_server.py` is a small Flask app that loads `models/best.pt` and exposes:

- `POST /classify` with `{ "image": "<base64 JPEG>" }`, returning class id, class name, confidence, and a `detected` flag
- `GET /health` for a liveness check

Unity captures a frame, posts it, and routes the arm based on the JSON reply.

### Web demo (Gradio)

Upload any image, tune the confidence slider, and get bounding boxes, per-item bin assignments, a Grad-CAM heatmap, and session statistics.

---

## Repository Structure

```
garbage-sorting-robot/
├── mrf_simulator.py          # 2D MRF simulator (OpenCV + YOLO)
├── inference_server.py       # Flask bridge for Unity
├── models/
│   └── best.pt               # YOLOv8s weights (~21.5 MB)
├── results_summary.txt       # Classifier metrics
├── yolo_results.txt          # Detector metrics
├── controllers/, worlds/     # Early Webots experiments (superseded)
└── README.md
```

The Unity project (`GarbageSortingRobot3D`) is kept separately because of its size. The datasets and training checkpoints are not stored in the repo.

---

## Getting Started

### Requirements

- Python 3.11 or newer
- macOS, Linux, or Windows
- Unity 6 with HDRP (only for the 3D scene)

### Install

```bash
git clone https://github.com/srividrumavangara/garbage-sorting-robot.git
cd garbage-sorting-robot
pip install ultralytics opencv-python numpy pillow flask
```

### Run the 2D simulator

Place some class-sorted sample images under `final_dataset/<class_name>/` (folders `wet_organic`, `recyclable_plastic`, `hazardous`, `health_medical`, `glass_metal`). Without them, items fall back to simulated detections.

```bash
python3 mrf_simulator.py
```

### Run the inference server

```bash
python3 inference_server.py
# listening on http://localhost:5000
curl http://localhost:5000/health
```

Then press Play in the Unity scene with the `AIClassifier` component pointed at that URL.

---

## Training Notes

Practical lessons from building this on free-tier GPUs:

- **Colab sessions reset often.** Back up datasets and checkpoints to Drive, and rebuild from the backup rather than re-downloading.
- **Kaggle T4 was more stable than Colab for long YOLO runs.** The Kaggle P100 failed with a CUDA-version mismatch against the pre-installed PyTorch, so pick the T4.
- **Never commit API tokens.** Use Colab Secrets or environment variables.
- **Unity/HDRP:** an HDRP-built asset imported into a URP project renders pink; start from an HDRP template. Keep an eye on disk space, since shader compilation is large.

---

## Current Status and Limitations

**Working**

- Both models trained and evaluated
- 2D MRF simulator with live YOLO inference
- Gradio demo
- Unity factory scene with items travelling on the conveyor by waypoints
- Flask bridge and Unity client script

**In progress**

- Final end-to-end hookup in Unity: classifier response driving the IK arm's pick-and-place

**Known limitations**

- Validation is simulation-only; no physical robot trials, so sim-to-real loss is unmeasured.
- Health/Medical has little training data (232 images) and no labels in the real-scene detection dataset.
- Training data includes some label noise (found through Grad-CAM).
- The Hazardous class partly relies on hazard symbols.
- The 3D scene currently uses coloured cubes as items rather than textured waste models.

---

## Roadmap

- Complete the Unity arm-to-classifier integration and record a demo
- Permanent hosting of the Gradio demo (for example, Hugging Face Spaces)
- Add medical-waste annotations to the detection dataset
- Clean mislabelled samples and retrain
- Sensor fusion for Hazardous items
- Hardware prototype (Jetson or Raspberry Pi with a low-cost arm) and sim-to-real evaluation
- ROS 2 / MoveIt 2 integration

---

## Acknowledgements

- **Supervisor:** V M Manikandan, Head of Department, CSE, SRM University AP
- Datasets: Kaggle `sumn2u/garbage-classification-v2`; Roboflow Universe YOLO Waste Detection
- Reference datasets and papers: TrashNet (Yang & Thung), TACO (Proença & Simões), Grad-CAM (Selvaraju et al.), domain randomization (Tobin et al.)
- Tools: PyTorch, Ultralytics YOLOv8, OpenCV, Gradio, Flask, Weights & Biases, Roboflow, Unity
- Factory environment: UnityFactorySceneHDRP by Unity Technologies Japan

## Author

**Sri Vidruma Vangara**, B.Tech CSE (AI/ML), SRM University AP
GitHub: [@srividrumavangara](https://github.com/srividrumavangara)

## License

Add a license of your choice (MIT is a common default for research code). Note that third-party datasets and the Unity asset are subject to their own licenses.
