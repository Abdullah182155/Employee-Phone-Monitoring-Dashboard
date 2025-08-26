import numpy as np

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0.0

def centroid_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return float(np.sqrt((c2_x - c1_x) ** 2 + (c2_y - c1_y) ** 2))

def is_active(phone_box: np.ndarray, body_box: np.ndarray, threshold: int) -> bool:
    x1 = max(phone_box[0], body_box[0])
    y1 = max(phone_box[1], body_box[1])
    x2 = min(phone_box[2], body_box[2])
    y2 = min(phone_box[3], body_box[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    phone_area = max(1, (phone_box[2] - phone_box[0]) * (phone_box[3] - phone_box[1]))
    return (inter_area / phone_area) >= (threshold / 100.0)

def merge_overlapping_boxes(
    boxes: np.ndarray, iou_threshold: float, distance_threshold: float
) -> np.ndarray:
    """Merge overlapping boxes using naive O(n^2) merge, grouping boxes that overlap or are close.
    Kept intentionally close to original algorithm but wrapped cleanly for reuse.
    """
    if boxes is None or len(boxes) <= 1:
        return boxes if boxes is not None else np.array([])

    boxes_list = [box.copy() for box in boxes]
    used = [False] * len(boxes_list)
    merged = []

    for i, b in enumerate(boxes_list):
        if used[i]:
            continue
        cur = b.copy()
        used[i] = True
        for j in range(i + 1, len(boxes_list)):
            if used[j]:
                continue
            iou = compute_iou(cur, boxes_list[j])
            dist = centroid_distance(cur, boxes_list[j])
            if iou >= iou_threshold or dist <= distance_threshold:
                # expand the bounding box
                cur[0] = min(cur[0], boxes_list[j][0])
                cur[1] = min(cur[1], boxes_list[j][1])
                cur[2] = max(cur[2], boxes_list[j][2])
                cur[3] = max(cur[3], boxes_list[j][3])
                used[j] = True
        merged.append(cur)

    return np.array(merged) if len(merged) > 0 else np.array([])