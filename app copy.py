import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
from ultralytics import YOLO
import torch
import os

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="📱 Employee Phone Monitoring", layout="wide")
st.title("📱 Employee Phone Monitoring Dashboard")

# -----------------------------
# Directory Setup
# -----------------------------
HISTORY_DIR = "history"
REPORT_DIR = "report"
HISTORY_FILE = os.path.join(HISTORY_DIR, "experiment_history.csv")

# Create directories if they don't exist
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Initialize history file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["Experiment ID", "Timestamp", "Source Type", "Total Phones", "Active Phones", "Inactive Phones", "CSV File"]).to_csv(HISTORY_FILE, index=False)

# -----------------------------
# Sidebar for Input Selection
# -----------------------------
st.sidebar.title("Input Source")
source_type = st.sidebar.radio("Choose input:", ("Camera", "Video"), index=1)  # Default to "Video"
if source_type == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])
else:
    uploaded_file = None

# Add Start Detection Button
start_detection = st.sidebar.button("Start Detection")

# Threshold Values
threshold = 90  # Static Active Phone Threshold set to 90
iou_match_threshold = 0.5  # IoU threshold for matching across frames
iou_merge_threshold = 0.6  # IoU threshold for merging overlapping detections
centroid_distance_threshold = 50  # Pixel distance threshold for merging based on centroid proximity

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
try:
    model_path = "best2.pt"  # Replace with your model path
    model = YOLO(model_path)
    model.fuse()
    if torch.cuda.is_available():
        model.to("cuda")
        try:
            model.model.half()
            st.info("Model converted to FP16 for faster inference.")
        except Exception as e:
            st.warning(f"FP16 not supported on this device: {e}. Using FP32.")
    else:
        st.warning("CUDA not available, falling back to CPU.")
        model.to("cpu")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# -----------------------------
# Helper Functions
# -----------------------------
def is_active(phone_box, body_box, threshold=90):
    """Check if phone overlaps body above threshold."""
    x1 = max(phone_box[0], body_box[0])
    y1 = max(phone_box[1], body_box[1])
    x2 = min(phone_box[2], body_box[2])
    y2 = min(phone_box[3], body_box[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    phone_area = max(1, (phone_box[2] - phone_box[0]) * (phone_box[3] - phone_box[1]))
    return inter_area / phone_area >= threshold / 100

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) for two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def compute_centroid_distance(box1, box2):
    """Compute Euclidean distance between centroids of two boxes."""
    c1_x = (box1[0] + box1[2]) / 2
    c1_y = (box1[1] + box1[3]) / 2
    c2_x = (box2[0] + box2[2]) / 2
    c2_y = (box2[1] + box2[3]) / 2
    return np.sqrt((c2_x - c1_x) ** 2 + (c2_y - c1_y) ** 2)

def merge_overlapping_boxes(boxes, iou_threshold, distance_threshold):
    """Merge overlapping boxes of the same class based on IoU and centroid distance."""
    if len(boxes) <= 1:
        return boxes
    merged_boxes = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        current_box = boxes[i]
        merged_box = current_box.copy()
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            iou = compute_iou(current_box, boxes[j])
            distance = compute_centroid_distance(current_box, boxes[j])
            if iou >= iou_threshold or distance <= distance_threshold:
                merged_box[0] = min(merged_box[0], boxes[j][0])
                merged_box[1] = min(merged_box[1], boxes[j][1])
                merged_box[2] = max(merged_box[2], boxes[j][2])
                merged_box[3] = max(merged_box[3], boxes[j][3])
                used[j] = True
        merged_boxes.append(merged_box)
    return np.array(merged_boxes)

# -----------------------------
# Tabs Setup
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Video Feed", "Current Report", "Experiment History"])

# -----------------------------
# Video Capture Setup
# -----------------------------
with tab1:
    if not start_detection:
        st.info("Please select an input source and click 'Start Detection' to begin.")
    elif source_type == "Video" and uploaded_file is None:
        st.warning("Please upload a video file!")
    elif start_detection:
        if source_type == "Video":
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                st.error("Cannot open camera. Please check your device.")
                st.stop()

        # Initialize Streamlit Elements
        stframe = st.empty()
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        phones_metric = metrics_col1.empty()
        active_metric = metrics_col2.empty()
        inactive_metric = metrics_col3.empty()

        # Tracking Setup
        tracked_phones = {}  # For active tracking
        all_tracked_phones = {}  # For persistent storage of all phones
        phone_id_counter = 0
        max_lost_frames = 10
        frame_counter = 0
        FRAME_SKIP = 4

        # Video Processing Loop
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1

            if frame_counter % FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (512, 640), interpolation=cv2.INTER_AREA)

            try:
                results = model(frame, conf=0.5, iou=0.5)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
            except Exception as e:
                st.warning(f"Error processing frame: {e}")
                continue

            # Merge overlapping detections for phones and bodies
            phones = merge_overlapping_boxes(boxes[classes == 1], iou_merge_threshold, centroid_distance_threshold)
            bodies = merge_overlapping_boxes(boxes[classes == 0], iou_merge_threshold, centroid_distance_threshold)

            current_frame_ids = []
            for p in phones:
                matched_id = None
                best_iou = 0
                best_distance = float('inf')
                for pid, info in tracked_phones.items():
                    iou = compute_iou(p, info["box"])
                    distance = compute_centroid_distance(p, info["box"])
                    if iou >= iou_match_threshold or (iou > 0.1 and distance < centroid_distance_threshold):
                        if iou > best_iou or (iou == best_iou and distance < best_distance):
                            matched_id = pid
                            best_iou = iou
                            best_distance = distance
                if matched_id is not None:
                    tracked_phones[matched_id]["last_seen"] = frame_counter
                    tracked_phones[matched_id]["box"] = p
                    all_tracked_phones[matched_id]["last_seen"] = frame_counter
                    all_tracked_phones[matched_id]["box"] = p
                    all_tracked_phones[matched_id]["end_time"] = datetime.now()
                    status = "Inactive"
                    for b in bodies:
                        if is_active(p, b, threshold):
                            status = "Active"
                            break
                    tracked_phones[matched_id]["status"] = status
                    all_tracked_phones[matched_id]["status"] = status
                    current_frame_ids.append(matched_id)
                else:
                    phone_id_counter += 1
                    status = "Inactive"
                    for b in bodies:
                        if is_active(p, b, threshold):
                            status = "Active"
                            break
                    phone_data = {
                        "status": status,
                        "start_time": datetime.now(),
                        "last_seen": frame_counter,
                        "box": p,
                        "end_time": datetime.now()
                    }
                    tracked_phones[phone_id_counter] = phone_data
                    all_tracked_phones[phone_id_counter] = phone_data.copy()
                    current_frame_ids.append(phone_id_counter)

            lost_ids = [pid for pid, info in tracked_phones.items() if frame_counter - info["last_seen"] > max_lost_frames]
            for pid in lost_ids:
                del tracked_phones[pid]  # Remove from active tracking, but keep in all_tracked_phones

            for pid, info in tracked_phones.items():
                x1, y1, x2, y2 = map(int, info["box"])
                color = (0, 255, 0) if info["status"] == "Active" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{info['status']}-{pid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            active_count = sum(1 for v in tracked_phones.values() if v["status"] == "Active")
            inactive_count = sum(1 for v in tracked_phones.values() if v["status"] == "Inactive")
            phones_metric.metric("📱 Phones", len(tracked_phones))
            active_metric.metric("🟢 Active", active_count)
            inactive_metric.metric("🔴 Inactive", inactive_count)

            stframe.image(cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

        # Generate and Save Report
        df_records = pd.DataFrame([
            {
                "ID": pid,
                "Status": info["status"],
                "Start Time": info["start_time"],
                "End Time": info["end_time"],
                "Duration": (info["end_time"] - info["start_time"]).total_seconds()
            } for pid, info in all_tracked_phones.items()
        ])

        # Save to history
        if not df_records.empty:
            report_filename = os.path.join(REPORT_DIR, f"phone_monitoring_report_{experiment_id}.csv")
            df_records.to_csv(report_filename, index=False)
            history_entry = pd.DataFrame([{
                "Experiment ID": experiment_id,
                "Timestamp": datetime.now(),
                "Source Type": source_type,
                "Total Phones": len(all_tracked_phones),
                "Active Phones": sum(1 for v in all_tracked_phones.values() if v["status"] == "Active"),
                "Inactive Phones": sum(1 for v in all_tracked_phones.values() if v["status"] == "Inactive"),
                "CSV File": report_filename
            }])
            history_df = pd.read_csv(HISTORY_FILE)
            history_df = pd.concat([history_df, history_entry], ignore_index=True)
            history_df.to_csv(HISTORY_FILE, index=False)
            st.success("✅ Video processing finished!")
        else:
            st.info("No phone detections recorded.")

# -----------------------------
# Current Report Tab
# -----------------------------
with tab2:
    st.header("Current Report")
    if 'df_records' in locals() and not df_records.empty:
        st.dataframe(df_records)
        st.download_button(
            label="💾 Download Current Report CSV",
            data=df_records.to_csv(index=False),
            file_name=f"phone_monitoring_report_{experiment_id}.csv",
            mime="text/csv"
        )
    else:
        st.info("No phone detections recorded in the current session.")

# -----------------------------
# History Tab
# -----------------------------
with tab3:
    st.header("Experiment History")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        if not history_df.empty:
            st.dataframe(history_df)
            for idx, row in history_df.iterrows():
                if os.path.exists(row["CSV File"]):
                    with open(row["CSV File"], "rb") as file:
                        st.download_button(
                            label=f"Download Report {row['Experiment ID']}",
                            data=file,
                            file_name=os.path.basename(row["CSV File"]),
                            mime="text/csv"
                        )
        else:
            st.info("No experiment history available.")
    else:
        st.info("No experiment history available.")