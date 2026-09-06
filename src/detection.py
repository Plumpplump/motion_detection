import cv2
import numpy as np

ANALYSIS_WIDTH = 128
DIFF_THRESHOLD = 25


def preprocess_frame(frame: np.ndarray, width: int = ANALYSIS_WIDTH) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aspect = frame.shape[1] / frame.shape[0]
    height = max(1, int(round(width / aspect)))
    small = cv2.resize(gray, (width, height))
    return cv2.GaussianBlur(small, (21, 21), 0)


def motion_ratio(current: np.ndarray, target: np.ndarray, threshold: int = DIFF_THRESHOLD) -> float:
    diff = cv2.absdiff(current, target)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(thresh)
    return changed / current.size