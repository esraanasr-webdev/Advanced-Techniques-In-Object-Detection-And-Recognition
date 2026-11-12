"""Run the first Smoke Before Fire occlusion pilot using the existing CNNs.

Place beside export_xai_predictions.py in the project root. Both checkpoints
stay fixed. The regions below are manually chosen equal-area image patches.
They cover selected parts of smoke/flame/background, not entire objects.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from export_xai_predictions import file_hash, load_module, match_predictions


IMAGE_NAME = "Frame_No1914_video_15_jpg.rf.955748aaf3d3b29c43a3d5ccb23d32ce.jpg"
# [left, top, right, bottom], right and bottom exclusive, at 1920 x 1080.
REGIONS = {
    "smoke_patch": [680, 130, 880, 330],
    "flame_patch": [800, 340, 1000, 540],
    "background_patch": [1360, 150, 1560, 350],
}


def prepare_images(image, output):
    from PIL import ImageDraw, ImageFilter, ImageStat

    if image.size != (1920, 1080):
        raise ValueError(f"Expected the original 1920 x 1080 image; received {image.size}.")
    fill = tuple(round(value) for value in ImageStat.Stat(image).mean)
    review = image.copy()
    draw = ImageDraw.Draw(review)
    variants = {}
    for name, box in REGIONS.items():
        draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], outline="magenta", width=4)
        draw.text((box[0], max(0, box[1] - 18)), name, fill="magenta", stroke_width=1)
        for method in ("mean_fill", "blur"):
            variant = image.copy()
            if method == "mean_fill":
                variant.paste(fill, tuple(box))
            else:
                patch = image.crop(tuple(box)).filter(ImageFilter.GaussianBlur(radius=30))
                variant.paste(patch, (box[0], box[1]))
            key = f"{name}_{method}"
            path = output / f"{key}.png"
            variant.save(path)
            variants[key] = path
    review.thumbnail((1280, 720))
    review.save(output / "regions_review.png")
    return variants, fill


def track_reference(reference, predictions, calculate_iou):
    """Track by maximum IoU with the original box, independently per reference."""
    best_overlap = 0.0
    best = None
    for prediction in predictions:
        overlap = float(calculate_iou(reference["box"], prediction["box"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best = prediction
    if best is None or best_overlap < 0.50:
        return {"tracking_status": "no_matching_box", "max_iou_to_original": best_overlap,
                "score": "", "score_change": ""}
    return {"tracking_status": "above_threshold" if best["confidence"] >= 0.05 else "below_threshold",
            "max_iou_to_original": best_overlap, "score": best["confidence"],
            "score_change": best["confidence"] - reference["confidence"]}


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_result_figure(output, image, variants, results, references):
    from PIL import Image, ImageDraw

    # The plot follows the first fire-matched Faster R-CNN reference.
    targets = [ref for ref in references["Faster_R_CNN"] if ref["fire_matched"]]
    if not targets:
        return
    reference = targets[0]
    rows = {row["variant"]: row for row in results
            if row["model"] == "Faster_R_CNN" and row["reference_id"] == reference["reference_id"]}
    canvas = Image.new("RGB", (1280, 4 * 404), "white")
    for index, key in enumerate(["original", *variants]):
        panel = image.copy() if key == "original" else Image.open(variants[key]).convert("RGB")
        draw = ImageDraw.Draw(panel)
        draw.rectangle(reference["box"], outline="blue", width=5)
        panel = panel.resize((640, 360))
        left, top = (index % 2) * 640, (index // 2) * 404
        canvas.paste(panel, (left, top + 44))
        draw = ImageDraw.Draw(canvas)
        row = rows[key]
        score = row["score"]
        score_text = f"score={score:.4f}" if score != "" else "score unavailable; no spatial match"
        draw.text((left + 6, top + 4), key, fill="black")
        draw.text((left + 6, top + 20), score_text + "; blue = ORIGINAL reference box", fill="black")
    canvas.save(output / "pilot_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Create patch previews without loading models.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    test = root / "dataset-a/test_dataset/Fire-and-Smoke-Detection-Dataset/Fire-and-Smoke-Detection-Dataset/dataset/test"
    image_path = args.image.resolve() if args.image else test / "images" / IMAGE_NAME
    if not image_path.is_file():
        raise FileNotFoundError(f"Original image missing: {image_path}")

    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if image.size != (1920, 1080):
        raise ValueError(f"Use the original image. Expected 1920 x 1080, received {image.size}.")
    export_root = args.output_root or root / "xai_occlusion"
    export_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_"), dir=export_root))
    variants, fill = prepare_images(image, output)
    metadata = {"completed": False, "image": str(image_path), "image_sha256": file_hash(image_path),
                "image_size": list(image.size), "regions": REGIONS, "patch_area_pixels": 40000,
                "selection": "manual exploratory patches; source image selected after baseline comparison",
                "fill_rgb": fill, "blur_radius": 30, "confidence": 0.05, "tracking_iou": 0.50,
                "interpretation": "Patch sensitivity for one selected image. Missing spatial matches have no measured score. Patches are approximate and contain surrounding context."}
    metadata_path = output / "pilot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared regions and variants in: {output}", flush=True)
    if args.prepare_only:
        return

    import torch
    import importlib.metadata

    training = root / "dataset-b/model_training"
    sources = [training / "yolo11n/evaluate_unseen_fire_yolo.py",
               training / "faster_rcnn/train.py",
               training / "faster_rcnn/evaluate_unseen_fire_faster_rcnn.py"]
    yolo = load_module("pilot_yolo_evaluator", sources[0])
    load_module("train", sources[1])
    faster = load_module("pilot_faster_evaluator", sources[2])
    checkpoints = (yolo.MODEL_PATH, faster.MODEL_PATH)
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    label_path = test / "labels" / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing fire annotation: {label_path}")
    gt = yolo.load_fire_boxes(label_path, image.width, image.height)
    if not gt:
        raise ValueError("The selected image has no fire annotations.")
    yolo_model = yolo.YOLO(str(yolo.MODEL_PATH))
    faster_model = faster.create_model()
    checkpoint = torch.load(faster.MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    faster_model.load_state_dict(state, strict=True)
    faster_model.to("cpu").eval()

    results = []
    references = {}
    counts = []
    with (output / "pilot_raw_predictions.jsonl").open("w", encoding="utf-8") as target:
        for key, path in {"original": image_path, **variants}.items():
            print(f"Running both models: {key}", flush=True)
            yolo_result = yolo_model.predict(source=str(path), conf=0.01, verbose=False, device="cpu")[0]
            raw = {"YOLO11n": [{"box": box.tolist(), "confidence": float(score)}
                               for box, score in zip(yolo_result.boxes.xyxy.cpu(), yolo_result.boxes.conf.cpu())]}
            with Image.open(path) as source:
                current_image = source.convert("RGB")
            with torch.no_grad():
                prediction = faster_model([faster.F.to_tensor(current_image)])[0]
            raw["Faster_R_CNN"] = [{"box": box.tolist(), "confidence": float(score)}
                                   for box, score in zip(prediction["boxes"].cpu(), prediction["scores"].cpu()) if float(score) >= 0.01]
            target.write(json.dumps({"variant": key, "predictions": raw}) + "\n")
            for name, predictions in raw.items():
                matched, tp = match_predictions(predictions, gt, yolo.calculate_iou)
                counts.append({"variant": key, "model": name, "predictions_at_005": len(matched), "fire_tp": tp})
                if key == "original":
                    ordered = sorted(matched, key=lambda row: (row["status"] != "TP", -row["score"]))[:5]
                    references[name] = [{"reference_id": index, "box": [row[axis] for axis in ("x1", "y1", "x2", "y2")],
                                         "confidence": row["score"], "fire_matched": row["status"] == "TP"}
                                        for index, row in enumerate(ordered)]
                    if name == "Faster_R_CNN" and tp != 1:
                        raise ValueError(f"Original Faster R-CNN image baseline changed: expected 1 fire TP, got {tp}.")
                    if name == "YOLO11n" and tp != 0:
                        raise ValueError(f"Original YOLO image baseline changed: expected 0 fire TP, got {tp}.")
                for reference in references[name]:
                    measured = track_reference(reference, predictions, yolo.calculate_iou)
                    results.append({"variant": key, "model": name, "reference_id": reference["reference_id"],
                                    "original_fire_matched": reference["fire_matched"],
                                    "original_score": reference["confidence"], **measured})
    write_rows(output / "pilot_results.csv", results)
    write_rows(output / "fire_overlap_counts.csv", counts)
    save_result_figure(output, image, variants, results, references)
    metadata.update({"completed": True, "original_image_baseline_matches": True, "references": references,
                     "checkpoints": {str(path): file_hash(path) for path in checkpoints},
                     "source_hashes": {str(path): file_hash(path) for path in [*sources, Path(__file__).resolve()]},
                     "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics")}})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nFaster R-CNN original fire-matched reference(s):")
    for row in results:
        if row["model"] == "Faster_R_CNN" and row["original_fire_matched"]:
            print(f"{row['variant']}: score={row['score']}, change={row['score_change']}, {row['tracking_status']}")
    print(f"\nSaved pilot_results.csv and pilot_comparison.png in: {output}")


if __name__ == "__main__":
    main()
