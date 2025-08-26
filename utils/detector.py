from typing import Optional, Tuple
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO
from .config import Config

class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.cfg.model_path)
            # attempt to fuse and convert to half if possible
            try:
                self.model.fuse()
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    self.model.to("cuda")
                    # attempt half precision
                    try:
                        self.model.model.half()
                    except Exception:
                        pass
                except Exception:
                    self.model.to("cpu")
            else:
                self.model.to("cpu")
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run model inference and return boxes and class indices.
        Returns:
            boxes: numpy array of xyxy boxes
            classes: numpy array of class indices (int)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        results = self.model(frame, conf=self.cfg.inference_conf, iou=self.cfg.inference_iou)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) and hasattr(results[0], "boxes") else np.array([])
        classes = (
            results[0].boxes.cls.cpu().numpy().astype(int) if len(results) and hasattr(results[0], "boxes") else np.array([])
        )
        return boxes, classes