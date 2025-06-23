# CrossViewID – Multi-Camera Player Tracking

> End-to-end pipeline that detects, tracks and **identifies the same player across two football camera feeds** (broadcast & tactical-cam) powered by YOLOv8 and deep SORT–style tracking.

---

## 1. Project Overview
CrossViewID automates the 3 classical steps of a multi-camera re-identification system:

1. **Detection** – Uses a YOLO model (`models/best.pt`) to find every player in each frame.
2. **Single-Camera Tracking** – Links detections through time into player tracks (Deep SORT logic).
3. **Cross-Camera Matching** – Matches tracks from the two cameras to output a *player-ID mapping*.

The pipeline is orchestrated by `main.py` and produces a JSON file summarising the results plus helpful console logs.

---

## 2. Repository Layout
```
Multi Camera Player Tracking
├── data/                # <-- sample videos go here
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── models/
│   └── best.pt          # YOLO player-detection weights
├── output/              # <-- created automatically, holds results JSON
├── utils/               # core modules (detector, tracker, matcher, helpers…)
├── main.py              # entry-point script
├── requirements.txt     # python dependencies
└── README.md            # you are here
```

---

## 3. Prerequisites
* **Python ≥ 3.9** (3.10/3.11 work fine)
* (Optional) **CUDA-enabled GPU** for faster inference.

---

## 4. Installation
```bash
# 1. Clone the repo
$ git clone https://github.com/sahniaditya007/CrossViewID-Multi-Camera-Player-Tracking.git
$ cd CrossViewID-Multi-Camera-Player-Tracking

# 2. Create and activate a virtual environment (recommended)
$ python -m venv .venv
$ source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install python packages
$ pip install -r requirements.txt
```


## 5. Quick Start
```bash
# just run the main script – all logs are printed to the console
$ python main.py
```

A results file such as `output/crossviewid_results_20250623_123456.json` will be written automatically.

### Running on CPU only
The script auto-detects CUDA. If no GPU is available it will fall back to CPU (consider lowering video resolution or length).

---

## 6. Custom Usage
The simplest workflow is baked into `main.py`. For more control import individual utilities:
```python
from utils.detector import run_detection
from utils.tracker import track_players
from utils.matcher import match_players_across_views
```
Each function is documented in-code and can be integrated into your own pipeline.

---

## 7. Troubleshooting
| Symptom | Possible Cause | Fix |
|---------|----------------|------|
| `Missing required files` error | Videos / weights not found | Check paths in the console message or edit `MODEL_DIR` / `DATA_DIR` in `main.py`. |
| CUDA requested but not available | Running on machine without GPU driver | Install CUDA toolkit & driver *or* use CPU. |
| Low match rate | Different halves, occlusions, poor lighting | Trim videos to overlapping period, fine-tune model, increase confidence threshold. |

---

## 8. Contributing
Pull requests are welcome! Please open an issue first to discuss major changes.

1. Fork the project & create your feature branch (`git checkout -b feat/amazing-feature`).
2. Commit your changes (`git commit -m 'feat: add amazing feature'`).
3. Push to the branch (`git push origin feat/amazing-feature`).
4. Open a PR.

---
