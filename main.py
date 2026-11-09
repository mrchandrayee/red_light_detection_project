
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from core.red_light import RedLightWatch, VehicleTrack

DEFAULT_CONFIG = {
    "light_polygon": [[145, 40], [220, 40], [220, 130], [145, 130]],
    "entry_polygon": [[70, 350], [300, 350], [350, 620], [20, 620]],
    "anchor": "top",
    "hold_frames": 3,
    "clear_frames": 5,
    "vehicle_classes": [2, 3, 5, 7],
    "confidence": 0.30
}

def load_config(path: str):
    p = Path(path)
    if not p.exists():
        p.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"Created {p}. Edit its polygons for your camera view.")
    return json.loads(p.read_text())

def main():
    parser = argparse.ArgumentParser(description="Red-light violation detection from a video/camera")
    parser.add_argument("--source", default="input.mp4", help="Video path, or camera index such as 0")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="output_annotated.mp4")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    model = YOLO(args.model)
    watcher = RedLightWatch(
        cfg["light_polygon"], cfg["entry_polygon"], cfg.get("anchor", "top"),
        cfg.get("hold_frames", 3), cfg.get("clear_frames", 5)
    )
    vehicle_classes = set(cfg.get("vehicle_classes", [2, 3, 5, 7]))
    violations = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                             conf=cfg.get("confidence", 0.30), verbose=False)[0]
        tracks = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            for tid, box, cls in zip(ids, boxes, classes):
                if cls in vehicle_classes:
                    tracks.append(VehicleTrack(tid, tuple(map(float, box))))

        is_red, lamp_boxes, fresh = watcher.update(frame, tracks)
        violations.update(fresh)

        light_color = (0, 0, 255) if is_red else (0, 220, 0)
        cv2.polylines(frame, [__import__("numpy").array(cfg["light_polygon"], dtype="int32")],
                      True, light_color, 2)
        cv2.polylines(frame, [__import__("numpy").array(cfg["entry_polygon"], dtype="int32")],
                      True, (255, 180, 0), 2)
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.box)
            violation = track.track_id in violations
            color = (0, 0, 255) if violation else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track.track_id}" + (" VIOLATION" if violation else ""),
                        (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for x1, y1, x2, y2 in lamp_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.putText(frame, f"Signal: {'RED' if is_red else 'NOT RED'} | Violations: {len(violations)}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, light_color, 2)
        writer.write(frame)
        if args.show:
            cv2.imshow("Red-light detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Annotated video: {args.output}")
    print(f"Unique track IDs flagged: {sorted(violations)}")

if __name__ == "__main__":
    main()
