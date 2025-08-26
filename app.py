import streamlit as st
from utils.config import Config
from utils.detector import Detector
from utils.tracker import PhoneTracker
from utils.processor import VideoProcessor
import tempfile
from datetime import datetime
import pandas as pd
from pathlib import Path

def run_app():
    cfg = Config()
    st.set_page_config(page_title=cfg.page_title, layout=cfg.layout)
    st.title(cfg.page_title)

    # Sidebar inputs
    st.sidebar.title("Input Source")
    source_type = st.sidebar.radio("Choose input:", ("Camera", "Video"), index=1)
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"]) if source_type == "Video" else None
    start_detection = st.sidebar.button("Start Detection")

    # Load model outside of main loop when user clicks start
    detector = None
    tracker = PhoneTracker(cfg)
    processor = None

    try:
        detector = Detector(cfg)
    except Exception:
        st.stop()

    processor = VideoProcessor(cfg, detector, tracker)

    tab1, tab2, tab3 = st.tabs(["Video Feed", "Current Report", "Experiment History"])

    # Main detection tab
    with tab1:
        if not start_detection:
            st.info("Please select an input source and click 'Start Detection' to begin.")
        elif source_type == "Video" and uploaded_file is None:
            st.warning("Please upload a video file!")
        else:
            # Prepare video source
            if source_type == "Video":
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.flush()
                source = tfile.name
                df_records, report_path = processor.process(source, use_camera=False)
            else:
                df_records, report_path = processor.process(0, use_camera=True)

            if not df_records.empty:
                st.success("✅ Video processing finished!")
            else:
                st.info("No phone detections recorded.")

    # Current report
    with tab2:
        st.header("Current Report")
        if 'df_records' in locals() and not df_records.empty:
            st.dataframe(df_records)
            st.download_button(
                label="💾 Download Current Report CSV",
                data=df_records.to_csv(index=False),
                file_name=f"phone_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No phone detections recorded in the current session.")

    # History
    with tab3:
        st.header("Experiment History")
        if cfg.history_file.exists():
            history_df = pd.read_csv(cfg.history_file)
            if not history_df.empty:
                st.dataframe(history_df)
                for idx, row in history_df.iterrows():
                    file_path = row.get("CSV File")
                    
            else:
                st.info("No experiment history available.")
        else:
            st.info("No experiment history available.")

if __name__ == "__main__":
    run_app()