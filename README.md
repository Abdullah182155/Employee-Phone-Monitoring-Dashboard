# 📱 Employee Phone Monitoring Dashboard

This project is a **real-time employee phone monitoring system** using computer vision. It detects phones and evaluates whether employees are actively using their phones based on the phone’s position relative to the body.

---

## Features

- Supports **video file upload** or **live camera feed**.
- Detects **phones and human bodies** using YOLOv8.
- Tracks phones across frames with **unique IDs**.
- Determines if a phone is **active** based on overlap with the body (fixed threshold: 90%).
- Merges overlapping phone or body detections to avoid duplicate counts.
- Generates **real-time metrics**: total phones, active phones, inactive phones.
- Stores session reports in CSV files.
- Maintains a **history of experiments** with download links for past reports.

---

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

2. Create a Python environment (recommended):

```bash
conda create -n phone_monitoring python=3.10
conda activate phone_monitoring
```

3. Install dependencies:

```bash
pip install streamlit opencv-python-headless ultralytics torch pandas numpy
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

---

### Dashboard Overview

The app has **three tabs**:

1. **Video Feed**  
   - Displays the live camera feed or uploaded video.
   - Shows bounding boxes for detected phones:
     - **Green**: Active phone  
     - **Red**: Inactive phone
   - Metrics:
     - Total phones detected
     - Active phones
     - Inactive phones

2. **Current Report**  
   - Shows a CSV table of phone detections for the current session.
   - Includes:
     - Phone ID
     - Status (Active/Inactive)
     - Start Time
     - End Time
     - Duration
   - Option to download the current report CSV.

3. **Experiment History**  
   - Displays all previous sessions.
   - Shows:
     - Experiment ID
     - Timestamp
     - Source type (Camera/Video)
     - Total phones, Active phones, Inactive phones
     - Download link for each report CSV

---

### How It Works

1. **Model Loading**  
   - YOLOv8 model (`best.pt`) is loaded and optimized for GPU (FP16 if available).  
   - If GPU is not available, CPU is used.

2. **Detection and Tracking**  
   - For each frame:
     - Detect phones and bodies.
     - Merge overlapping boxes.
     - Assign unique IDs to phones and track them across frames.
     - Determine phone activity:
       - **Active if phone overlaps body >= 90%**
     - Update phone status in real-time metrics.

3. **Merging and Matching**  
   - Overlapping boxes are merged based on IoU and centroid distance.
   - Phones are matched across frames to maintain consistent IDs.

4. **Report Generation**  
   - Generates CSV report for each session.
   - Saves experiment history in `history/experiment_history.csv`.
   - Provides download links for reports in the app.

---

### Directory Structure

```
phone_monitoring/
├── app.py                     # Streamlit entry point
├── best2.pt                   # YOLOv8 model weights (rename to models/best.pt or update config)
├── Dockerfile                 # Docker configuration for containerization
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── utils/
│   ├── __init__.py
│   ├── config.py              # Configuration for thresholds, file paths, and model path
│   ├── detector.py            # YOLOv8 wrapper for object detection
│   ├── tracker.py             # PhoneTracker class for tracking logic
│   ├── processor.py           # Video and camera processing logic
│   └── helpers.py             # Utility functions (IoU, box merging, etc.)
├── models/                    # Create this directory and move best2.pt to best.pt inside it
├── history/                   # Stores experiment history CSV (auto-created)
└── report/                    # Stores generated session reports (auto-created)
```

---

### Code Changes for Fixed Threshold

In `app.py`, replace the threshold slider with a fixed value:

```python
# Remove slider
# threshold = st.sidebar.slider("Active Phone Threshold (%)", 50, 100, 90)

# Use static threshold
threshold = 90
```

The function `is_active(phone_box, body_box, threshold)` now always uses `threshold = 90`.

---

### Notes

- The app is optimized to **skip frames** for faster processing (`FRAME_SKIP = 4` by default).
- Maximum lost frames for tracking is set to 10.
- YOLO class IDs:
  - `0`: Body
  - `1`: Phone

---

### Requirements

- streamlit==1.38.0
- torch==2.7.0+cu118
- torchvision==0.22.0+cu118
- torchaudio==2.7.0+cu118
- ultralytics==8.3.186
- opencv-python==4.10.0.84
- opencv-contrib-python==4.11.0.86
- numpy==1.26.4
- pandas==2.2.2

---

### License

This project is **open-source** and free to use for educational and research purposes.
