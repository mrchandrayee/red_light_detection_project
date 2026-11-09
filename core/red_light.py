
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import cv2
import numpy as np

# Reconstructed from the visible code. Tune these values for the camera.
MIN_RED = 110
RED_MARGIN = 12
MIN_AREA = 80
MIN_SIDE = 10
MAX_SIDE = 80
MIN_FILL = 0.45
MIN_CIRCULARITY = 0.55
MIN_ASPECT = 0.60

Point = tuple[int, int]
Polygon = Sequence[Point]
Box = tuple[float, float, float, float]


def find_red_lamps(frame: np.ndarray, polygon: Polygon) -> list[Box]:
    """Find compact, filled red objects inside a polygon (frame coordinates)."""
    if len(polygon) < 3:
        return []
    shape = np.asarray(polygon, dtype=np.int32)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shape], 255)

    blue, green, red = cv2.split(frame)
    red16 = red.astype(np.int16)
    binary = (
        (red16 > MIN_RED)
        & (red16 > green.astype(np.int16) + RED_MARGIN)
        & (red16 > blue.astype(np.int16) + RED_MARGIN)
        & (mask > 0)
    ).astype(np.uint8) * 255

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lamps: list[Box] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        if area < MIN_AREA or width < MIN_SIDE or height < MIN_SIDE:
            continue
        if width > MAX_SIDE or height > MAX_SIDE:
            continue
        fill = area / float(width * height)
        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0 if perimeter <= 0 else 4.0 * np.pi * area / (perimeter * perimeter)
        aspect = min(width, height) / float(max(width, height))
        if fill < MIN_FILL or circularity < MIN_CIRCULARITY or aspect < MIN_ASPECT:
            continue
        top = (x + width / 2.0, float(y))
        if cv2.pointPolygonTest(shape, top, False) <= 0:
            continue
        lamps.append((float(x), float(y), float(x + width), float(y + height)))
    return lamps


def anchor(box: Box, mode: str = "top") -> tuple[float, float]:
    x1, y1, x2, y2 = box
    if mode == "center":
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    if mode == "bottom":
        return ((x1 + x2) / 2, y2)
    return ((x1 + x2) / 2, y1)


@dataclass
class VehicleTrack:
    track_id: int
    box: Box
    lost: bool = False


class RedLightWatch:
    """Remember lamp state and count each track once when it enters the entry zone."""
    def __init__(self, light: Polygon, entry: Polygon, anchor_mode: str = "top",
                 hold_frames: int = 3, clear_frames: int = 5):
        self.light = list(light)
        self.entry = list(entry)
        self.anchor_mode = anchor_mode
        self.hold_frames = max(1, int(hold_frames))
        self.clear_frames = max(1, int(clear_frames))
        self.is_red = False
        self.red_run = 0
        self.clear_run = 0
        self.inside: dict[int, bool] = {}
        self.prev_y: dict[int, float] = {}
        self.entered: set[int] = set()

    def reset(self) -> None:
        self.is_red = False
        self.red_run = self.clear_run = 0
        self.inside.clear()
        self.prev_y.clear()
        self.entered.clear()

    def update(self, frame: np.ndarray, tracks: Iterable[VehicleTrack]):
        lamps = find_red_lamps(frame, self.light)
        if lamps:
            self.red_run += 1
            self.clear_run = 0
            if self.red_run >= self.hold_frames:
                self.is_red = True
        else:
            self.red_run = 0
            self.clear_run += 1
            if self.clear_run >= self.clear_frames:
                self.is_red = False

        fresh: list[int] = []
        alive: set[int] = set()
        shape = np.asarray(self.entry, dtype=np.int32)
        for track in tracks:
            if track.lost:
                continue
            tid = int(track.track_id)
            alive.add(tid)
            point = anchor(track.box, self.anchor_mode)
            previous_y = self.prev_y.get(tid)
            self.prev_y[tid] = point[1]
            inside = cv2.pointPolygonTest(shape, point, False) >= 0
            was_inside = self.inside.get(tid)
            self.inside[tid] = inside
            moving_up = previous_y is not None and point[1] < previous_y
            if was_inside is not False or not inside or not self.is_red or not moving_up:
                continue
            if tid in self.entered:
                continue
            self.entered.add(tid)
            fresh.append(tid)

        self.inside = {tid: value for tid, value in self.inside.items() if tid in alive}
        self.prev_y = {tid: y for tid, y in self.prev_y.items() if tid in alive}
        return self.is_red, lamps, fresh
