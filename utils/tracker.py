from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import numpy as np
import cv2
from .config import Config
from .helpers import compute_iou, centroid_distance, is_active

@dataclass
class TrackedPhone:
    id: int
    box: np.ndarray
    status: str
    start_time: datetime
    last_seen: int
    end_time: datetime = field(default_factory=datetime.now)

class PhoneTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracked: Dict[int, TrackedPhone] = {}
        self.all_tracked: Dict[int, TrackedPhone] = {}
        self._id_counter = 0

    def _new_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def update_with_detections(self, phones: np.ndarray, bodies: np.ndarray, frame_counter: int) -> List[int]:
        """Match detections to existing tracked phones and update/create tracked entries.
        Returns list of current frame IDs.
        """
        current_ids: List[int] = []

        # for each detected phone, find best match
        for p in phones:
            matched = None
            best_iou = 0.0
            best_dist = float("inf")

            for pid, info in self.tracked.items():
                iou_val = compute_iou(p, info.box)
                dist_val = centroid_distance(p, info.box)
                if iou_val >= self.cfg.iou_match_threshold or (iou_val > 0.1 and dist_val < self.cfg.centroid_distance_threshold):
                    if iou_val > best_iou or (iou_val == best_iou and dist_val < best_dist):
                        matched = pid
                        best_iou = iou_val
                        best_dist = dist_val

            status = "Inactive"
            for b in bodies:
                if is_active(p, b, self.cfg.active_overlap_threshold):
                    status = "Active"
                    break

            if matched is not None:
                tp = self.tracked[matched]
                tp.box = p
                tp.last_seen = frame_counter
                tp.status = status
                tp.end_time = datetime.now()
                self.all_tracked[matched] = tp
                current_ids.append(matched)
            else:
                new_id = self._new_id()
                tp = TrackedPhone(
                    id=new_id,
                    box=p,
                    status=status,
                    start_time=datetime.now(),
                    last_seen=frame_counter,
                    end_time=datetime.now(),
                )
                self.tracked[new_id] = tp
                self.all_tracked[new_id] = tp
                current_ids.append(new_id)

        # remove lost
        lost = [pid for pid, info in self.tracked.items() if frame_counter - info.last_seen > self.cfg.max_lost_frames]
        for pid in lost:
            if pid in self.tracked:
                del self.tracked[pid]

        return current_ids

    def draw_annotations(self, frame: np.ndarray) -> None:
        for pid, info in self.tracked.items():
            x1, y1, x2, y2 = map(int, info.box)
            color = (0, 255, 0) if info.status == "Active" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{info.status} {pid}", (x1, y1 - 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)