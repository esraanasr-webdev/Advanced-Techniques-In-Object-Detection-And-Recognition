"""Check the unmodified PNG control and prepare a fixed additional XAI sample.

Place beside the three previous XAI scripts in the project root. The sample
is selected from image filenames, independently of model scores. One image
per known video ID is selected, excluding the original pilot's video group.
New sample images receive no model inference in this step.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

from export_xai_predictions import file_hash, load_module
from run_occlusion_pilot import IMAGE_NAME
from run_sliding_occlusion import infer


def image_group(name):
    video = re.search(r"video_(\d+)", name, flags=re.IGNORECASE)
    if video:
        return "video_" + video.group(1), "filename_video_id"
    return "image_" + name.split(".rf.")[0], "filename_only"


def rank_key(value, seed):
    return hashlib.sha256((seed + "\n" + value).encode("utf-8")).hexdigest()


def select_sample(images, excluded_name, count, seed):
    excluded_group = image_group(excluded_name)[0]
    groups = {}
    for path in images:
        group, basis = image_group(path.name)
        if group == excluded_group:
            continue
        groups.setdefault(group, []).append((path, basis))
    ordered = sorted(groups, key=lambda group: rank_key(group, seed))
    selected = []
    for group in ordered[:count]:
        path, basis = min(groups[group], key=lambda item: rank_key(item[0].name, seed))
        selected.append((path, group, basis))
    return selected


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_sample(output, selected, label_directory):
    from PIL import Image, ImageDraw, ImageOps

    columns = 4
    rows_per_page = 3
    page_size = columns * rows_per_page
    records = []
    sheet = None
    for index, (path, group, basis) in enumerate(selected):
        label = label_directory / f"{path.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing sample annotation: {label}")
        counts = {0: 0, 1: 0}
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5 or int(parts[0]) not in counts:
                raise ValueError(f"Unexpected annotation row in {label}: {line}")
            counts[int(parts[0])] += 1
        sample_id = f"sample_{index + 1:02d}"
        records.append({"sample_id": sample_id, "image": path.name, "image_path": str(path),
                        "source_group": group, "group_basis": basis, "image_sha256": file_hash(path),
                        "label_sha256": file_hash(label), "fire_gt_count": counts[0], "smoke_gt_count": counts[1],
                        "scene_type": "pending", "visible_smoke": "pending", "visible_flame": "pending",
                        "usable_for_xai": "pending", "review_notes": ""})
        local = index % page_size
        if local == 0:
            sheet = Image.new("RGB", (1920, 1080), "white")
        with Image.open(path) as source:
            thumbnail = ImageOps.contain(source.convert("RGB"), (470, 270))
        left = (local % columns) * 480
        top = (local // columns) * 360
        sheet.paste(thumbnail, (left + (480 - thumbnail.width) // 2, top + 72))
        draw = ImageDraw.Draw(sheet)
        draw.text((left + 8, top + 8), f"{sample_id} | {group[:44]}", fill="black")
        draw.text((left + 8, top + 26), f"annotation boxes: fire={counts[0]}, smoke={counts[1]}", fill="black")
        draw.text((left + 8, top + 44), "Selected by filenames; review visibility manually", fill="black")
        if local == page_size - 1 or index == len(selected) - 1:
            sheet.save(output / f"validation_contact_sheet_{index // page_size + 1}.png")
    if not records:
        raise ValueError("No additional source groups available for sampling.")
    write_csv(output / "validation_manifest.csv", records, list(records[0]))
    return records


def compare_predictions(original, control, calculate_iou):
    """Greedy spatial pairing, followed by score and coordinate tolerances."""
    remaining = set(range(len(control)))
    rows = []
    for index, prediction in enumerate(original):
        best = None
        overlap = 0.0
        for candidate in sorted(remaining):
            current = float(calculate_iou(prediction["box"], control[candidate]["box"]))
            if current > overlap:
                best, overlap = candidate, current
        if best is None or overlap < 0.99:
            rows.append({"reference_id": index, "original_score": prediction["confidence"],
                         "png_score": "", "score_change": "", "box_iou": overlap,
                         "max_coordinate_change_pixels": "", "within_tolerance": False})
            continue
        remaining.remove(best)
        score_change = control[best]["confidence"] - prediction["confidence"]
        coordinate_change = max(abs(a - b) for a, b in zip(prediction["box"], control[best]["box"]))
        rows.append({"reference_id": index, "original_score": prediction["confidence"],
                     "png_score": control[best]["confidence"], "score_change": score_change,
                     "box_iou": overlap, "max_coordinate_change_pixels": coordinate_change,
                     "within_tolerance": abs(score_change) <= 1e-5 and coordinate_change <= 1e-3})
    passed = len(original) == len(control) and not remaining and all(row["within_tolerance"] for row in rows)
    return rows, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sample-size", type=int, default=12)
    args = parser.parse_args()
    if args.sample_size < 1:
        raise ValueError("sample-size must be positive.")
    root = args.project_root.resolve()
    training = root / "dataset-b/model_training"
    source_paths = [training / "yolo11n/evaluate_unseen_fire_yolo.py", training / "faster_rcnn/train.py",
                    training / "faster_rcnn/evaluate_unseen_fire_faster_rcnn.py"]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing source {path}. Put this script in the project root.")

    import cv2
    import numpy as np
    import torch
    import importlib.metadata
    from PIL import Image

    yolo = load_module("validation_yolo_evaluator", source_paths[0])
    load_module("train", source_paths[1])
    faster = load_module("validation_faster_evaluator", source_paths[2])
    if yolo.TEST_IMAGES.resolve() != faster.TEST_IMAGES.resolve() or yolo.TEST_LABELS.resolve() != faster.TEST_LABELS.resolve():
        raise ValueError("Evaluator test paths differ.")
    image_path = yolo.TEST_IMAGES / IMAGE_NAME
    for path in (image_path, yolo.MODEL_PATH, faster.MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
    images = sorted(path for path in yolo.TEST_IMAGES.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    seed = "smoke-before-fire-validation-v1"
    selected = select_sample(images, IMAGE_NAME, args.sample_size, seed)
    export_root = root / "xai_validation"
    export_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_"), dir=export_root))
    records = prepare_sample(output, selected, yolo.TEST_LABELS)
    metadata = {"completed": False, "pilot_image": str(image_path), "pilot_sha256": file_hash(image_path),
                "sample_seed": seed, "requested_sample_size": args.sample_size, "selected_sample_size": len(records),
                "excluded_pilot_group": image_group(IMAGE_NAME)[0],
                "selection": "One image per filename video group when present; hash-based order; no prediction scores used. Filename-only grouping does not establish independent events.",
                "score_tolerance": 1e-5, "coordinate_tolerance_pixels": 1e-3,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics")},
                "checkpoints": {str(path): file_hash(path) for path in (yolo.MODEL_PATH, faster.MODEL_PATH)},
                "sources": {str(path): file_hash(path) for path in
                            [*source_paths, Path(__file__).resolve(),
                             Path(__file__).with_name("run_sliding_occlusion.py"),
                             Path(__file__).with_name("run_occlusion_pilot.py"),
                             Path(__file__).with_name("export_xai_predictions.py")]}}
    summary_path = output / "validation_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Fixed sample: {len(records)} images. Output directory: {output}", flush=True)
    control_path = output / "unmodified_png_control.png"
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.save(control_path)
    with Image.open(control_path) as control:
        pil_equal = np.array_equal(np.asarray(image), np.asarray(control.convert("RGB")))
    jpeg_cv = cv2.imread(str(image_path))
    png_cv = cv2.imread(str(control_path))
    if jpeg_cv is None or png_cv is None or jpeg_cv.shape != png_cv.shape:
        raise ValueError("Image decoding failed or dimensions differ.")
    cv_difference = np.abs(jpeg_cv.astype(np.int16) - png_cv.astype(np.int16))
    metadata["pixels"] = {"PIL_equal": bool(pil_equal), "OpenCV_equal": bool(np.array_equal(jpeg_cv, png_cv)),
                           "OpenCV_max_channel_difference": int(cv_difference.max()),
                           "OpenCV_mean_channel_difference": float(cv_difference.mean())}
    yolo_model = yolo.YOLO(str(yolo.MODEL_PATH))
    faster_model = faster.create_model()
    checkpoint = torch.load(faster.MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    faster_model.load_state_dict(state, strict=True)
    faster_model.to("cpu").eval()
    print("Running the original JPG...", flush=True)
    original = infer(image_path, yolo_model, faster_model, faster)
    print("Running the unmodified PNG...", flush=True)
    control = infer(control_path, yolo_model, faster_model, faster)
    (output / "noop_raw_predictions.json").write_text(json.dumps({"original_jpg": original, "unmodified_png": control}, indent=2), encoding="utf-8")
    per_box = []
    comparisons = {}
    for name, predictions in original.items():
        rows, passed = compare_predictions(predictions, control[name], yolo.calculate_iou)
        per_box.extend({"model": name, **row} for row in rows)
        comparisons[name] = {"within_tolerance": passed, "original_returned_boxes": len(predictions),
                             "png_returned_boxes": len(control[name]),
                             "original_boxes_at_005": sum(p["confidence"] >= 0.05 for p in predictions),
                             "png_boxes_at_005": sum(p["confidence"] >= 0.05 for p in control[name])}
    fields = ["model", "reference_id", "original_score", "png_score", "score_change", "box_iou", "max_coordinate_change_pixels", "within_tolerance"]
    write_csv(output / "noop_comparison.csv", per_box, fields)
    metadata.update({"completed": True, "prediction_comparisons": comparisons,
                     "noop_passed": bool(pil_equal and np.array_equal(jpeg_cv, png_cv)
                                         and all(item["within_tolerance"] for item in comparisons.values()))})
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"pixels": metadata["pixels"], "models": comparisons, "NOOP_PASSED": metadata["noop_passed"]}, indent=2))
    print(f"Review validation_contact_sheet_1.png and validation_manifest.csv in: {output}")
    if not metadata["noop_passed"]:
        print("Inspect the measured differences before interpreting new occlusion runs. The fixed sample is still available for manual review.")


if __name__ == "__main__":
    main()
