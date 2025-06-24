# CrossViewID: Multi-Camera Player Tracking

CrossViewID is a lightweight, end-to-end pipeline for automatically detecting, tracking **and** matching football (soccer) players that appear simultaneously in two time-synchronised videos – for example a television *broadcast* feed and a high-angle *tacticam* feed. It is designed to be:

* **Modular** – detection, single-camera tracking and cross-view matching are fully decoupled and can be swapped out independently.
* **Hardware-aware** – will seamlessly use CUDA if available, otherwise falls back to CPU execution.
* **Easy to run** – a single `python main.py` launches the complete workflow and produces a self-contained JSON report.

<p align="center">
  <img src="docs/sample_pipeline.png" width="720" alt="Cross-camera tracking pipeline"/>
</p>

---

## Features

1. **YOLO-v8 Detection** – fast person / player detection driven by Ultralytics YOLO weights (`models/best.pt`).
2. **IoU-based Tracking** – simple yet effective per-camera tracker with configurable IoU and track-length thresholds.
3. **Spatial-Temporal Matching** – greedy / Hungarian matching that leverages centre-point distance & frame overlap.
4. **JSON Reporting** – human- and machine-readable results saved to `output/crossviewid_results_<timestamp>.json`.
5. **Extensive Logging** – progress and statistics are logged to the console for each pipeline stage.

---

## Repository Layout

```
├── data/               # Expected location of input videos
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── models/             # YOLO weights (e.g. best.pt)
├── output/             # Auto-generated results will appear here
├── utils/              # Detector, tracker and matcher modules
│   ├── detector.py
│   ├── tracker.py
│   └── matcher.py
├── main.py             # Pipeline entry-point
├── requirements.txt    # Python dependencies
└── README.md           # (this file)
```

---

## Installation

CrossViewID requires **Python ≥ 3.9**. Create a virtual environment (recommended) and install the dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU users**: Replace the `torch` line in `requirements.txt` with the CUDA build that matches your driver, or follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## Preparing Assets

1. **Model weights** – Download or train a YOLO-v8 model that recognises players and place it at `models/best.pt`.
2. **Input videos** – Copy the *time-synchronised* videos into `data/broadcast.mp4` and `data/tacticam.mp4`.

> The pipeline assumes the videos are roughly aligned in time. If they are not, you may need to trim or shift them beforehand.

---

## Quick Start

```bash
python main.py
```

The script will:

1. **Check** required files exist.
2. **Detect** players frame-by-frame in each video.
3. **Track** each player through their respective view.
4. **Match** identities across views.
5. **Save** a results file such as `output/crossviewid_results_20250624_112515.json`.

A console summary similar to the following will also be displayed:

```
============================================================
CROSSVIEWID RESULTS
============================================================
Broadcast video: 18 player tracks
Tacticam video: 20 player tracks
Cross-camera matches: 16

 Player ID Mapping:
   Tacticam #3 → Broadcast #2
   Tacticam #7 → Broadcast #5
   ...

📊 Match rate: 80.0%
============================================================
```

---

## Result File Schema

The exported JSON contains:

```jsonc
{
  "timestamp": "YYYYMMDD_HHMMSS",
  "config": {
    "device": "cpu|cuda",
    "model_path": "models/best.pt",
    "broadcast_video": "data/broadcast.mp4",
    "tacticam_video": "data/tacticam.mp4"
  },
  "statistics": {
    "broadcast_tracks": 18,
    "tacticam_tracks": 20,
    "matched_players": 16,
    "match_rate": 0.8
  },
  "player_mapping": {
    "<tacticam_track_id>": <broadcast_track_id>
  }
}
```

---

## Customisation & Tips

* **Thresholds** – Tweak IoU / lost-frame / min-length thresholds inside `utils/tracker.py`.
* **Matching Strategy** – Switch between `greedy` and `hungarian` algorithms in `utils/matcher.py` or directly in `main.py`.
* **Progress Intervals** – All core modules expose `progress_interval` parameters for more or fewer log messages.
* **Large Resolutions** – If you hit out-of-memory errors, down-sample frames before detection or reduce YOLO input size.

---

## Roadmap

* Support for **multi-match fusion** across >2 cameras.
* Plug-in hooks for **deep re-identification**.
* Optional **GUI / stream visualiser**.

Contributions are welcome – see below!

---

## Contributing

1. Fork the repo and create your branch: `git checkout -b feature/awesome`.
2. Commit your changes: `git commit -m 'Add awesome feature'`.
3. Push to the branch: `git push origin feature/awesome`.
4. Open a pull request.

Please format code with `black` and ensure new tests pass.

---

## License

This project is released under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* Ultralytics YOLO-v8 \- <https://github.com/ultralytics/ultralytics>
* NumPy, SciPy, OpenCV and PyTorch projects.

---

## Contact

For questions or commercial licensing, please open an issue or email **your.email@example.com**.
