from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

@dataclass
class Config:
    page_title: str = "📱 Employee Phone Monitoring"
    layout: str = "wide"
    history_dir: Path = Path("history")
    report_dir: Path = Path("history/report")
    history_file: Path = field(init=False)

    # thresholds (unchanged semantics)
    active_overlap_threshold: int = 90  # percent
    iou_match_threshold: float = 0.5
    iou_merge_threshold: float = 0.6
    centroid_distance_threshold: float = 50

    # detection / tracking
    frame_skip: int = 1
    max_lost_frames: int = 10

    # model
    model_path: str = "models/best.pt"
    inference_conf: float = 0.5
    inference_iou: float = 0.5

    def __post_init__(self):
        self.history_file = self.history_dir / "experiment_history.csv"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            pd.DataFrame(
                columns=[
                    "Experiment ID",
                    "Timestamp",
                    "Source Type",
                    "Total Phones",
                    "Active Phones",
                    "Inactive Phones",
                    "CSV File",
                ]
            ).to_csv(self.history_file, index=False)