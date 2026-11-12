"""Sliding-window sensitivity maps for the existing Smoke Before Fire pilot.

Place beside export_xai_predictions.py and run_occlusion_pilot.py in the
project root. Default patches are 200x200 pixels, stride 200, with mean fill
and blur. Maps describe one original prediction at a time. Missing box
matches are recorded separately from measured score changes.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from export_xai_predictions import file_hash, load_module, match_predictions
from run_occlusion_pilot import IMAGE_NAME, track_reference, write_rows


def positions(length, window, stride):
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_windows(width, height, window, stride):
    if not 0 < stride <= window <= min(width, height):
        raise ValueError("Use 0 < stride <= window <= min(image width, image height).")
    return [[x, y, x + window, y + window]
            for y in positions(height, window, stride)
            for x in positions(width, window, stride)]


def perturb(image, box, method, fill):
    from PIL import ImageFilter

    changed = image.copy()
    if method == "mean_fill":
        changed.paste(fill, tuple(box))
    else:
        patch = image.crop(tuple(box)).filter(ImageFilter.GaussianBlur(radius=30))
        changed.paste(patch, (box[0], box[1]))
    return changed


def infer(path, yolo_model, faster_model, faster):
    import torch
    from PIL import Image

    result = yolo_model.predict(source=str(path), conf=0.01, verbose=False, device="cpu")[0]
    raw = {"YOLO11n": [{"box": box.tolist(), "confidence": float(score)}
                       for box, score in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu())]}
    with Image.open(path) as source:
        image = source.convert("RGB")
    with torch.no_grad():
        result = faster_model([faster.F.to_tensor(image)])[0]
    raw["Faster_R_CNN"] = [{"box": box.tolist(), "confidence": float(score)}
                           for box, score in zip(result["boxes"].cpu(), result["scores"].cpu())
                           if float(score) >= 0.01]
    return raw


def sensitivity_maps(rows, width, height):
    import numpy as np

    score_sum = np.zeros((height, width), dtype=np.float32)
    matched_count = np.zeros((height, width), dtype=np.int32)
    lost_count = np.zeros((height, width), dtype=np.int32)
    coverage = np.zeros((height, width), dtype=np.int32)
    for row in rows:
        x1, y1, x2, y2 = (row[key] for key in ("x1", "y1", "x2", "y2"))
        region = (slice(y1, y2), slice(x1, x2))
        coverage[region] += 1
        if row["tracking_status"] == "no_matching_box":
            lost_count[region] += 1
        else:
            score_sum[region] += row["score_drop"]
            matched_count[region] += 1
    score_drop = np.full((height, width), np.nan, dtype=np.float32)
    lost_fraction = np.full((height, width), np.nan, dtype=np.float32)
    np.divide(score_sum, matched_count, out=score_drop, where=matched_count > 0)
    np.divide(lost_count, coverage, out=lost_fraction, where=coverage > 0)
    return score_drop, lost_fraction, coverage


def render_maps(output, image, references, rows, methods):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    for model, model_references in references.items():
        for reference in model_references:
            selected = [row for row in rows if row["model"] == model and row["reference_id"] == reference["reference_id"]]
            extent = max([abs(row["score_drop"]) for row in selected if row["score_drop"] != ""] + [0.001])
            figure, axes = plt.subplots(len(methods), 3, figsize=(16, 4.3 * len(methods)), squeeze=False)
            for index, method in enumerate(methods):
                method_rows = [row for row in selected if row["method"] == method]
                score_drop, lost_fraction, coverage = sensitivity_maps(method_rows, image.width, image.height)
                stem = f"{model}_ref_{reference['reference_id']}_{method}"
                np.savez_compressed(output / f"{stem}.npz", matched_score_drop=score_drop,
                                    no_match_fraction=lost_fraction, coverage=coverage)
                for axis in axes[index]:
                    axis.imshow(image)
                    axis.set_axis_off()
                box = reference["box"]
                axes[index, 0].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                                 edgecolor="blue", facecolor="none", linewidth=2))
                axes[index, 0].set_title(f"Original reference {reference['reference_id']}; score {reference['confidence']:.4f}\n{method}")
                heat = axes[index, 1].imshow(np.ma.masked_invalid(score_drop), cmap="RdBu_r", vmin=-extent, vmax=extent, alpha=0.70)
                axes[index, 1].set_title("Score drop when a box match exists\nred = decrease, blue = increase, clear = no measurement")
                figure.colorbar(heat, ax=axes[index, 1], fraction=0.03, pad=0.02)
                heat = axes[index, 2].imshow(np.ma.masked_invalid(lost_fraction), cmap="magma", vmin=0, vmax=1, alpha=0.70)
                axes[index, 2].set_title("Fraction of covering windows with no spatial match\n1 = lost in every covering perturbation")
                figure.colorbar(heat, ax=axes[index, 2], fraction=0.03, pad=0.02)
            label = "matched fire GT" if reference["fire_matched"] else "unmatched to fire GT"
            figure.suptitle(f"{model}, original reference {reference['reference_id']} ({label})", fontsize=14)
            figure.text(0.5, 0.01, "Exploratory patch sensitivity on one image. Each pixel averages effects of windows covering it; this is not pixel-level attribution.", ha="center", fontsize=9)
            figure.tight_layout(rect=(0, 0.04, 1, 0.95))
            figure.savefig(output / f"{model}_ref_{reference['reference_id']}_heatmaps.png", dpi=140)
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    args = parser.parse_args()
    root = args.project_root.resolve()
    training = root / "dataset-b/model_training"
    yolo_source = training / "yolo11n/evaluate_unseen_fire_yolo.py"
    train_source = training / "faster_rcnn/train.py"
    faster_source = training / "faster_rcnn/evaluate_unseen_fire_faster_rcnn.py"
    for path in (yolo_source, train_source, faster_source):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Put the script in the project root.")

    import torch
    import importlib.metadata
    from PIL import Image, ImageStat

    yolo = load_module("sliding_yolo_evaluator", yolo_source)
    load_module("train", train_source)
    faster = load_module("sliding_faster_evaluator", faster_source)
    if yolo.TEST_IMAGES.resolve() != faster.TEST_IMAGES.resolve() or yolo.TEST_LABELS.resolve() != faster.TEST_LABELS.resolve():
        raise ValueError("The two evaluators point to different test inputs.")
    image_path = yolo.TEST_IMAGES / IMAGE_NAME
    label_path = yolo.TEST_LABELS / f"{Path(IMAGE_NAME).stem}.txt"
    for path in (image_path, label_path, yolo.MODEL_PATH, faster.MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input {path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    windows = make_windows(image.width, image.height, args.window, args.stride)
    methods = ("mean_fill", "blur")
    fill = tuple(round(value) for value in ImageStat.Stat(image).mean)
    gt = yolo.load_fire_boxes(label_path, image.width, image.height)
    yolo_model = yolo.YOLO(str(yolo.MODEL_PATH))
    faster_model = faster.create_model()
    checkpoint = torch.load(faster.MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    faster_model.load_state_dict(state, strict=True)
    faster_model.to("cpu").eval()

    original = infer(image_path, yolo_model, faster_model, faster)
    references = {}
    expected = {"YOLO11n": (1, 0), "Faster_R_CNN": (2, 1)}
    for model, predictions in original.items():
        matched, tp = match_predictions(predictions, gt, yolo.calculate_iou)
        if (len(matched), tp) != expected[model]:
            raise ValueError(f"Original image baseline changed for {model}: predictions={len(matched)}, fire TP={tp}. Expected {expected[model]}.")
        ordered = sorted(matched, key=lambda row: (row["status"] != "TP", -row["score"]))[:5]
        references[model] = [{"reference_id": index, "box": [row[key] for key in ("x1", "y1", "x2", "y2")],
                              "confidence": row["score"], "fire_matched": row["status"] == "TP"}
                             for index, row in enumerate(ordered)]
    export_root = root / "xai_sliding"
    export_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_"), dir=export_root))
    metadata = {"completed": False, "image": str(image_path), "image_sha256": file_hash(image_path),
                "labels_sha256": file_hash(label_path), "window": args.window, "stride": args.stride,
                "window_count": len(windows), "methods": methods, "fill_rgb": fill, "blur_radius": 30,
                "reporting_confidence": 0.05, "inference_confidence": 0.01, "tracking_iou": 0.50,
                "device": "cpu", "references": references,
                "checkpoints": {str(path): file_hash(path) for path in (yolo.MODEL_PATH, faster.MODEL_PATH)},
                "sources": {str(path): file_hash(path) for path in
                            (yolo_source, train_source, faster_source, Path(__file__).resolve(),
                             Path(__file__).with_name("export_xai_predictions.py"),
                             Path(__file__).with_name("run_occlusion_pilot.py"))},
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics")},
                "interpretation": "Window sensitivity for selected original boxes. Missing matches are separate from score changes. Maps average covering windows and describe one post-selected image."}
    summary_path = output / "sliding_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Original baseline matches. Windows: {len(windows)}; methods: {len(methods)}", flush=True)
    print(f"Output directory: {output}", flush=True)
    rows = []
    with (output / "sliding_raw_predictions.jsonl").open("w", encoding="utf-8") as raw_file, tempfile.TemporaryDirectory(prefix="input_", dir=output) as temporary:
        raw_file.write(json.dumps({"method": "original", "predictions": original}) + "\n")
        input_path = Path(temporary) / "window.png"
        for method in methods:
            for index, box in enumerate(windows):
                perturb(image, box, method, fill).save(input_path)
                predictions = infer(input_path, yolo_model, faster_model, faster)
                raw_file.write(json.dumps({"method": method, "window_id": index, "window": box, "predictions": predictions}) + "\n")
                raw_file.flush()
                for model, model_references in references.items():
                    for reference in model_references:
                        measured = track_reference(reference, predictions[model], yolo.calculate_iou)
                        drop = -measured["score_change"] if measured["score_change"] != "" else ""
                        rows.append({"method": method, "window_id": index, "x1": box[0], "y1": box[1],
                                     "x2": box[2], "y2": box[3], "model": model,
                                     "reference_id": reference["reference_id"], "original_fire_matched": reference["fire_matched"],
                                     "original_score": reference["confidence"], **measured, "score_drop": drop})
                if (index + 1) % 10 == 0 or index + 1 == len(windows):
                    print(f"{method}: {index + 1}/{len(windows)} windows", flush=True)
                write_rows(output / "sliding_results.csv", rows)
    print("Rendering heatmaps...", flush=True)
    render_maps(output, image, references, rows, methods)
    metadata["completed"] = True
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed. Send sliding_results.csv and Faster_R_CNN_ref_0_heatmaps.png from: {output}")


if __name__ == "__main__":
    main()
