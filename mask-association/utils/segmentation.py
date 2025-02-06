from collections import Counter
from typing import List, Dict, Tuple, Optional, Set

import torch
from torchtyping import TensorType as TorchTensor

import numpy as np
import cv2


Mask = TorchTensor[..., "H", "W"]
BBox = TorchTensor[..., 4]


def convert_bbox_format(bbox: BBox, conversion: str) -> BBox:
    """
    Convert bounding box formats between
    - LTRB: (left, top, right, bottom)
    - LTWH: (left, top, width, height)
    - CCWH: (center_x, center_y, width, height)
    """
    def lrwh2ltrb(bbox: BBox) -> BBox:
        l, r, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        return torch.stack([l, r, l + w, r + h], dim=-1)
    
    def ltrb2lrwh(bbox: BBox) -> BBox:
        l, t, r, b = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        return torch.stack([l, t, r - l, b - t], dim=-1)
    
    def lrwh2ccwh(bbox: BBox) -> BBox:
        l, r, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        return torch.stack([l + w / 2, r + h / 2, w, h], dim=-1)
    
    def ccwh2lrwh(bbox: BBox) -> BBox:
        x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        return torch.stack([x - w / 2, y - h / 2, w, h], dim=-1)
    
    def ltrb2ccwh(bbox: BBox) -> BBox:
        l, t, r, b = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        return torch.stack([(l + r) / 2, (t + b) / 2, r - l, b - t], dim=-1)
    
    def ccwh2ltrb(bbox: BBox) -> BBox:
        x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)
    
    return eval(conversion)(bbox)


def expand_bbox(bbox: BBox, scale: float, image_dim: Tuple) -> BBox:
    """
    Expand bounding box by `scale` factor.
    """
    x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_w  = (x2 - x1) * scale
    new_h  = (y2 - y1) * scale
    new_x1 = max(center_x - new_w / 2, 0)
    new_y1 = max(center_y - new_h / 2, 0)
    new_x2 = min(center_x + new_w / 2, image_dim[1])
    new_y2 = min(center_y + new_h / 2, image_dim[0])
    return torch.tensor([new_x1, new_y1, new_x2, new_y2])


def crop(image: TorchTensor["H", "W", "C"], bbox: BBox) -> TorchTensor["h", "w", "C"]:
    """
    Crop image according to bounding box in LTRB format.
    """
    x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
    return image[int(y1):int(y2), int(x1):int(x2)]


def points_in_bbox(pnts: TorchTensor["N", 2], bbox: BBox, return_indices=False) -> TorchTensor["M", 2]:
    """
    Given a set of points `pnts` (x, y) and a LRTB bounding box, return subset of points that lie within.
    """
    x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
    indices = torch.nonzero(
        (pnts[:, 0] >= x1) & 
        (pnts[:, 0] <= x2) &
        (pnts[:, 1] >= y1) & 
        (pnts[:, 1] <= y2)
    ).reshape(-1)
    return indices if return_indices else pnts[indices]


def points_in_bmask(pnts: TorchTensor["N", 2], bmask: Mask, return_indices=False, dialate=False):
    """
    Given a set of points `pnts` (x, y) and a binary mask, return subset of points that lie within, or None if no points
    """
    if dialate:
        bmask = dialate_bmask(bmask)
    if len(pnts) == 0:
        return pnts
    pnts = pnts.to(torch.long)
    mask = bmask[pnts[:, 1], pnts[:, 0]] == 1
    if return_indices:
        return torch.nonzero(mask).reshape(-1)
    else:
        return pnts[mask]
    

def iou_bbox(bbox1: BBox, bbox2: BBox) -> float:
    """
    Compute IoU score between two bounding boxes in LTBR format.
    """
    l1, t1, r1, b1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    l2, t2, r2, b2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    intersection = max(0, min(r1, r2) - max(l1, l2)) * max(0, min(b1, b2) - max(t1, t2))
    union = (r1 - l1) * (b1 - t1) + (r2 - l2) * (b2 - t2) - intersection
    return intersection / union


def iou_bmask(bmask1: Mask, bmask2: Mask) -> float:
    """
    Compute IoU score between two binary masks.
    """
    intersection = (bmask1 & bmask2).sum()
    union        = (bmask1 | bmask2).sum()
    return intersection / union


def deduplicate_bmasks(bmasks: Mask, iou=0.85, return_indices=False) -> Mask:
    """
    Given binary masks `bmasks`, deduplicate them as defined by IoU threshold.
    """
    bmasks_dedup, indices = [bmasks[0]], [0]
    for i, bmask in enumerate(bmasks[1:]):
        ious = [iou_bmask(bmask, bm) for bm in bmasks_dedup]
        if max(ious) < iou:
            bmasks_dedup.append(bmask), indices.append(i + 1)
    bmasks_dedup, indices = torch.stack(bmasks_dedup), torch.tensor(indices)
    if return_indices:
        return bmasks_dedup, indices
    return bmasks_dedup


def deduplicate_bboxes(bboxes: BBox, iou=0.85, return_indices=False) -> BBox:
    """
    Given bounding boxes `bboxes` in LRTB format, deduplicate them as defined by IoU threshold.
    """
    bboxes_dedup, indices = [bboxes[0]], [0]
    for i, bbox in enumerate(bboxes[1:]):
        ious = [iou_bbox(bbox, bb) for bb in bboxes_dedup]
        if max(ious) < iou:
            bboxes_dedup.append(bbox), indices.append(i + 1)
    bboxes_dedup, indices = torch.stack(bboxes_dedup), torch.tensor(indices)
    if return_indices:
        return bboxes_dedup, indices
    return bboxes_dedup


def dialate_bmask(bmask: Mask, kernel_size=5) -> Mask:
    """
    Dialate each binary mask in `bmask` by `kernel_size` pixels.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    bmask = bmask.numpy().astype(np.uint8)
    bmask = cv2.dilate(bmask, kernel, iterations=1)
    return torch.from_numpy(bmask)


def assign_bmasks(bmasks: Mask, labels: TorchTensor) -> Mask:
    """
    Given binary masks `bmasks` and their `labels`, assign each pixel in each mask to its respective label.
    """
    bmasks_assigned = torch.zeros_like(bmasks)
    for i, bmask in enumerate(bmasks):
        bmasks_assigned[bmask.to(torch.bool)] = labels[i]
    return bmasks_assigned


def assign_mask(m: Mask, conversions: Dict, ignore_labels: Optional[Set]=None) -> Mask:
    """
    Assign mask labels according to `conversions`.
    """
    ignore_labels = ignore_labels or set()
    m_updated = torch.zeros_like(m).to(m)
    for label1, label2 in conversions.items():
        if label1 in ignore_labels: 
            continue
        m_updated[m == label1] = label2
    return m_updated


def combine_bmasks(bmasks: Mask, order=None) -> Mask:
    """
    Given binary masks `bmasks`, combine them into a single mask, optionally sorted by area.
    """
    assert order in [None, 'ascending', 'descending']
    if order is not None:
        bmasks = sorted(bmasks, key=lambda bmask: bmask.sum(), reverse=(order == 'descending'))
    combined_mask = torch.zeros(bmasks[0].shape, dtype=torch.int32)
    for i, bmask in enumerate(bmasks):
        combined_mask[bmask.to(torch.bool)] = i + 1
    return combined_mask


def decompose_mask(mask: Mask, return_labels=False) -> List[Mask]:
    """
    Decompose mask into individual bmasks.
    """
    labels = torch.unique(mask[mask > 0])
    bmasks = []
    for label in labels:
        bmasks.append(mask == label)
    return (bmasks, labels) if return_labels else bmasks


def sample_bmask(bmask: Mask, k: int, dialate=False) -> TorchTensor["k", 2]:
    """
    Sample `k` points from `bmask`. If there are more points than `k`, then returns all points.
    """
    if dialate:
        bmask = dialate_bmask(bmask)
    indices = torch.nonzero(bmask)
    return indices[torch.randperm(len(indices))[:k]]


def sample_mask(
    mask: Mask, k: int, dialate=False, ignore_labels: Optional[Set]=None, return_flattened=False
) -> Dict[int, TorchTensor["k", 2]]:
    """
    Sample `k` points per region from `mask` using sample_bmask().
    """
    ignore_labels = ignore_labels or set()
    label2points = {}
    for label in torch.unique(mask):
        if label.item() in ignore_labels: 
            continue
        indices = sample_bmask(mask == label, k, dialate=dialate)
        label2points[label.item()] = indices
    if return_flattened:
        return label2points, torch.cat([v for v in label2points.values()])
    return label2points


def remove_artifacts(mask: Mask, mode: str, min_area=128) -> Mask:
    """
    Removes small islands/fill holes from a mask.
    """
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')

    def remove_helper(bmask):
        bmask = (mode_holes ^ bmask).astype(np.uint8)
        nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
        sizes = stats[:, -1][1:]  # Row 0 corresponds to 0 pixels
        fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
        if not mode_holes:
            fill = [i for i in range(nregions) if i not in fill]
        return np.isin(regions, fill)

    mask = mask.numpy()
    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask): # opencv connected components operates on binary masks only
        mask_combined[remove_helper(mask == label)] = label
    return torch.from_numpy(mask_combined)


def mask_sequence_stats(m: Mask, ignore_labels: Optional[Set]=None):
    """
    Compute label statistics for a sequence of masks, including:
    - number of frames each label appears in
    - total area of each label
    - mean  area of each label
    """
    ignore_labels = ignore_labels or set()
    track_count = Counter()
    areas       = Counter()
    areas_mean  = Counter()
    for frame in m:
        labels = torch.unique(frame).tolist()
        for label in labels:
            if label in ignore_labels: 
                continue
            track_count[label] += 1
            areas[label] += torch.sum(frame == label).item()
    areas_mean = {k: v / track_count[k] for k, v in areas.items()}
    return track_count, areas, areas_mean


def match_masks(
    m1: Mask,
    m2: Mask,
    m1_labelcount: Optional[int]=None,
    m2_labelcount: Optional[int]=None,
    ignore_labels: Optional[Set]=None,
    return_conversions=False,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Tuple[TorchTensor, Dict]:
    """
    Match region by finding the most overlapping label in m2 for each label in m1.

    :param m1_labels: labels corresponding to each mask in `m1`, defaults to unique labels in m1
    :param m1_labelcount: number of labels in m1, defaults to number of unique labels in m1
    :param m2_labelcount: number of labels in m2, defaults to number of unique labels in m2
    :param ignore_labels: set of labels to ignore i.e. background
    :param return_conversions: whether to return the conversions betweem domains of m1 and m2
    """
    assert m1.shape == m2.shape
    ignore_labels = ignore_labels or set()
    m1 = m1.to(device)
    m2 = m2.to(device)

    labels1 = torch.unique(m1)
    labels2 = torch.unique(m2)
    labels2_domain_size = len(labels2) if m2_labelcount is None else m2_labelcount
    m12 = m1 * labels2_domain_size + m2 # potentially broadcast
    
    counts = torch.bincount(m12.flatten())
    conversions = {}
    for label1 in labels1:
        if label1 in ignore_labels: continue
        lb = label1 * labels2_domain_size
        ub = label1 * labels2_domain_size + labels2_domain_size
        label2 = torch.argmax(counts[lb.item():ub.item()])
        conversions[label1.item()] = label2.item()
    
    m = torch.zeros_like(m1)
    for label1, label2 in conversions.items():
        if label1 in ignore_labels: continue
        m[m1 == label1] = label2
    return (m, conversions) if return_conversions else m

