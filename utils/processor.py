from datetime import datetime
import pandas as pd
import cv2
import streamlit as st
import numpy as np
from .config import Config
from .detector import Detector
from .tracker import PhoneTracker
from .helpers import merge_overlapping_boxes

class VideoProcessor:
    def __init__(self, cfg: Config, detector: Detector, tracker: PhoneTracker):
        self.cfg = cfg
        self.detector = detector
        self.tracker = tracker

    def process(self, source: str, use_camera: bool = False) -> tuple[pd.DataFrame, str]:
        """Run processing loop. Returns dataframe of records and experiment_id (filename suffix).
        This keeps the same behaviour and final report structure as the original script.
        """
        # open capture
        cap = cv2.VideoCapture(0 if use_camera else source, cv2.CAP_DSHOW if use_camera else 0)
        if not cap.isOpened():
            st.error("Cannot open video source.")
            return pd.DataFrame(), ""

        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_counter = 0

        stframe = st.empty()
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        phones_metric = metrics_col1.empty()
        active_metric = metrics_col2.empty()
        inactive_metric = metrics_col3.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1

            if frame_counter % self.cfg.frame_skip != 0:
                continue

            frame = cv2.resize(frame, (512, 640), interpolation=cv2.INTER_AREA)

            try:
                boxes, classes = self.detector.detect(frame)
            except Exception as e:
                st.warning(f"Error processing frame: {e}")
                continue

            # split according to class indices (assumed same meaning as original: 1->phone, 0->body)
            phones = boxes[classes == 1] if len(boxes) and len(classes) else np.array([])
            bodies = boxes[classes == 0] if len(boxes) and len(classes) else np.array([])

            phones = merge_overlapping_boxes(phones, self.cfg.iou_merge_threshold, self.cfg.centroid_distance_threshold)
            bodies = merge_overlapping_boxes(bodies, self.cfg.iou_merge_threshold, self.cfg.centroid_distance_threshold)

            _ = self.tracker.update_with_detections(phones, bodies, frame_counter)
            self.tracker.draw_annotations(frame)

            active_count = sum(1 for v in self.tracker.tracked.values() if v.status == "Active")
            inactive_count = sum(1 for v in self.tracker.tracked.values() if v.status == "Inactive")
            phones_metric.metric("📱 Phones", len(self.tracker.tracked))
            active_metric.metric("🟢 Active", active_count)
            inactive_metric.metric("🔴 Inactive", inactive_count)

            stframe.image(cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

        # build df
        df_records = pd.DataFrame([
            {
                "ID": pid,
                "Status": info.status,
                "Start Time": info.start_time,
                "End Time": info.end_time,
                "Duration": (info.end_time - info.start_time).total_seconds(),
            }
            for pid, info in self.tracker.all_tracked.items()
        ])

        # save report
        if not df_records.empty:
            report_filename = str(self.cfg.report_dir / f"phone_monitoring_report_{experiment_id}.csv")
            df_records.to_csv(report_filename, index=False)

            # update history
            history_entry = pd.DataFrame([
                {
                    "Experiment ID": experiment_id,
                    "Timestamp": datetime.now(),
                    "Source Type": "Camera" if use_camera else "Video",
                    "Total Phones": len(self.tracker.all_tracked),
                    "Active Phones": sum(1 for v in self.tracker.all_tracked.values() if v.status == "Active"),
                    "Inactive Phones": sum(1 for v in self.tracker.all_tracked.values() if v.status == "Inactive"),
                    "CSV File": report_filename,
                }
            ])
            history_df = pd.read_csv(self.cfg.history_file)
            history_df = pd.concat([history_df, history_entry], ignore_index=True)
            history_df.to_csv(self.cfg.history_file, index=False)

            return df_records, report_filename

        return pd.DataFrame(), ""