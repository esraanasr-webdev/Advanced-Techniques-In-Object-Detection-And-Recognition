"""Export the existing YOLO/Faster R-CNN baseline for an XAI pilot.

Place this script in the project root and run with the existing yolo_env Python.
It uses the repository's model loaders and inference settings. Outputs go into
a new timestamped xai_exports directory. It performs no training.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile


def load_module(name, path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def match_predictions(predictions, ground_truth, calculate_iou):
    """Same score-ordered greedy matching as the existing CNN evaluators."""
    used = set()
    rows = []
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        if prediction["confidence"] < 0.05:
            continue
        best_iou = 0.0
        best_index = None
        for index, box in enumerate(ground_truth):
            if index in used:
                continue
            overlap = float(calculate_iou(prediction["box"], box))
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        matched = best_index is not None and best_iou >= 0.50
        if matched:
            used.add(best_index)
        rows.append({
            "score": prediction["confidence"],
            "x1": prediction["box"][0], "y1": prediction["box"][1],
            "x2": prediction["box"][2], "y2": prediction["box"][3],
            "status": "TP" if matched else "FP",
            "matched_fire_gt_index": best_index if matched else "",
            "best_unmatched_fire_iou": best_iou,
        })
    return rows, len(used)


def save_preview(path, image_path, ground_truth, model_rows):
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    scale = 640 / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = Image.new("RGB", (1280, size[1] + 48), "white")
    for panel, (name, rows) in enumerate(model_rows.items()):
        offset = panel * 640
        canvas.paste(image.resize(size), (offset, 48))
        draw = ImageDraw.Draw(canvas)
        draw.text((offset + 6, 4), name, fill="black")
        draw.text((offset + 6, 20), "green=fire GT, blue=TP, red=FP; up to 5 predictions, TP first", fill="black")
        for box in ground_truth:
            draw.rectangle([offset + box[0] * scale, 48 + box[1] * scale,
                            offset + box[2] * scale, 48 + box[3] * scale], outline="green", width=2)
        ordered = sorted(rows, key=lambda row: (row["status"] != "TP", -row["score"]))
        for row in ordered[:5]:
            box = [offset + row["x1"] * scale, 48 + row["y1"] * scale,
                   offset + row["x2"] * scale, 48 + row["y2"] * scale]
            color = "blue" if row["status"] == "TP" else "red"
            draw.rectangle(box, outline=color, width=2)
            draw.text((max(offset, box[0]), max(48, box[1])), f"{row['status']} {row['score']:.3f}", fill=color)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    training = root / "dataset-b/model_training"
    yolo_source = training / "yolo11n/evaluate_unseen_fire_yolo.py"
    faster_source = training / "faster_rcnn/evaluate_unseen_fire_faster_rcnn.py"
    train_source = training / "faster_rcnn/train.py"
    for source in (yolo_source, faster_source, train_source):
        if not source.is_file():
            raise FileNotFoundError(f"Missing source {source}. Place this script in the project root.")

    import torch
    from PIL import Image

    yolo = load_module("xai_yolo_evaluator", yolo_source)
    load_module("train", train_source)
    faster = load_module("xai_faster_evaluator", faster_source)
    if yolo.TEST_IMAGES.resolve() != faster.TEST_IMAGES.resolve() or yolo.TEST_LABELS.resolve() != faster.TEST_LABELS.resolve():
        raise ValueError("The existing evaluators point to different test sets. Resolve this before exporting.")
    for path in (yolo.MODEL_PATH, faster.MODEL_PATH, yolo.TEST_IMAGES, yolo.TEST_LABELS):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")
    images = sorted(path for path in yolo.TEST_IMAGES.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        raise ValueError("No test images found.")
    missing = [path.stem for path in images if not (yolo.TEST_LABELS / f"{path.stem}.txt").is_file()]
    if missing:
        raise ValueError(f"Missing label files for {len(missing)} images; first examples: {missing[:5]}")

    yolo_model = yolo.YOLO(str(yolo.MODEL_PATH))
    faster_model = faster.create_model()
    checkpoint = torch.load(faster.MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    faster_model.load_state_dict(state, strict=True)
    faster_model.to("cpu").eval()

    export_root = root / "xai_exports"
    export_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_"), dir=export_root))
    print(f"Export directory: {output}", flush=True)
    totals = {name: {"images": 0, "fire_images": 0, "gt_boxes": 0, "predictions": 0,
                     "tp": 0, "fp": 0, "fn": 0, "detected_images": 0}
              for name in ("YOLO11n", "Faster_R_CNN")}
    image_rows = []
    candidates = []
    box_fields = ["image", "model", "score", "x1", "y1", "x2", "y2", "status", "matched_fire_gt_index", "best_unmatched_fire_iou"]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as box_file, (output / "raw_predictions.jsonl").open("w", encoding="utf-8") as raw_file:
        box_writer = csv.DictWriter(box_file, fieldnames=box_fields)
        box_writer.writeheader()
        for position, image_path in enumerate(images, 1):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            label_path = yolo.TEST_LABELS / f"{image_path.stem}.txt"
            gt = yolo.load_fire_boxes(label_path, image.width, image.height)
            yolo_result = yolo_model.predict(source=str(image_path), conf=0.01, verbose=False, device="cpu")[0]
            raw = {"YOLO11n": [{"box": box.tolist(), "confidence": float(score)}
                               for box, score in zip(yolo_result.boxes.xyxy.cpu(), yolo_result.boxes.conf.cpu())]}
            with torch.no_grad():
                result = faster_model([faster.F.to_tensor(image)])[0]
            raw["Faster_R_CNN"] = [{"box": box.tolist(), "confidence": float(score)}
                                   for box, score in zip(result["boxes"].cpu(), result["scores"].cpu()) if float(score) >= 0.01]
            raw_file.write(json.dumps({"image": image_path.name, "image_sha256": file_hash(image_path),
                                       "labels_sha256": file_hash(label_path), "fire_gt_boxes": gt,
                                       "predictions": raw}) + "\n")
            matched = {}
            per_model = {}
            for name, predictions in raw.items():
                rows, tp = match_predictions(predictions, gt, yolo.calculate_iou)
                matched[name] = rows
                per_model[name] = tp
                box_writer.writerows({"image": image_path.name, "model": name, **row} for row in rows)
                counts = {"images": 1, "fire_images": int(bool(gt)), "gt_boxes": len(gt),
                          "predictions": len(rows), "tp": tp, "fp": len(rows) - tp,
                          "fn": len(gt) - tp, "detected_images": int(tp > 0)}
                image_rows.append({"image": image_path.name, "model": name, **counts})
                for key, value in counts.items():
                    totals[name][key] += value
            if per_model["Faster_R_CNN"] > 0 and per_model["YOLO11n"] == 0:
                candidates.append({"image": image_path.name, "faster_tp": per_model["Faster_R_CNN"], "yolo_tp": 0})
                if len(candidates) <= 3:
                    save_preview(output / f"candidate_{len(candidates)}.png", image_path, gt, matched)
            if position % 50 == 0 or position == len(images):
                print(f"Processed {position}/{len(images)}", flush=True)

    write_csv(output / "per_image.csv", image_rows)
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["image", "faster_tp", "yolo_tp"])
        writer.writeheader()
        writer.writerows(candidates)
    expected = {"YOLO11n": {"predictions": 1628, "tp": 6, "fp": 1622, "fn": 989, "detected_images": 6},
                "Faster_R_CNN": {"predictions": 2856, "tp": 51, "fp": 2805, "fn": 944, "detected_images": 51}}
    baseline_matches = all(counts["images"] == 637 and counts["fire_images"] == 459 and counts["gt_boxes"] == 995
                           and all(counts[key] == value for key, value in expected[name].items())
                           for name, counts in totals.items())
    metadata = {"completed": True, "confidence": 0.05, "iou": 0.50, "device": "cpu",
                "baseline_matches": baseline_matches, "totals": totals, "expected": expected,
                "candidate_count": len(candidates), "test_images": str(yolo.TEST_IMAGES),
                "test_labels": str(yolo.TEST_LABELS),
                "checkpoints": {str(path): file_hash(path) for path in (yolo.MODEL_PATH, faster.MODEL_PATH)},
                "source_hashes": {str(path): file_hash(path) for path in (yolo_source, faster_source, train_source)},
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics")}}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))
    print(f"BASELINE MATCHES: {baseline_matches}")
    print(f"Candidate images: {len(candidates)}. Preview files show the first three by filename; inspect smoke/flame visibility manually.")
    print(f"Outputs saved to: {output}")
    if not baseline_matches:
        raise SystemExit("Counts differ from the saved baseline. Inspect summary.json before interpreting the candidate images.")


if __name__ == "__main__":
    main()
