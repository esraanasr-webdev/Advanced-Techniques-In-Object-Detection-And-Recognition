"""Overlay validation annotations and summarize cached predictions.

Place beside export_xai_predictions.py in the project root. No model is run.
The existing 12-image selection is preserved. Fire and smoke matching are
reported as separate annotation-overlap tasks and must not be summed.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from export_xai_predictions import file_hash, match_predictions


def box_iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_annotations(path, width, height):
    annotations = {"fire": [], "smoke": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 or int(parts[0]) not in (0, 1):
            raise ValueError(f"Unexpected annotation row in {path}: {line}")
        cx, cy, bw, bh = map(float, parts[1:])
        name = "fire" if int(parts[0]) == 0 else "smoke"
        annotations[name].append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                                  (cx + bw / 2) * width, (cy + bh / 2) * height])
    return annotations


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("xai_exports/20261111_090800_6siaomz1"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    baseline = args.baseline_dir if args.baseline_dir.is_absolute() else root / args.baseline_dir
    with manifest_path.open(encoding="utf-8") as source:
        manifest = list(csv.DictReader(source))
    if not manifest:
        raise ValueError("Validation manifest is empty.")
    names = [row["image"] for row in manifest]
    if len(set(names)) != len(names):
        raise ValueError("The manifest contains duplicate image entries.")
    metadata = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if not metadata.get("completed") or not metadata.get("baseline_matches"):
        raise ValueError("Use the completed export that reproduced the baseline.")
    selected_names = set(names)
    cached = {}
    with (baseline / "raw_predictions.jsonl").open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["image"] in selected_names:
                if record["image"] in cached:
                    raise ValueError(f"Duplicate cached record: {record['image']}")
                cached[record["image"]] = record
    if selected_names != set(cached):
        raise ValueError(f"Missing cached images: {sorted(selected_names - set(cached))}")
    from PIL import Image, ImageDraw, ImageOps

    export_root = root / "xai_validation_audit"
    export_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_"), dir=export_root))
    image_directory = Path(metadata["test_images"])
    label_directory = Path(metadata["test_labels"])
    review_rows = []
    baseline_rows = []
    all_refs = []
    for index, item in enumerate(manifest):
        image_path = image_directory / item["image"]
        label_path = label_directory / f"{image_path.stem}.txt"
        record = cached[item["image"]]
        image_hash = file_hash(image_path)
        label_hash = file_hash(label_path)
        if image_hash != item["image_sha256"] or image_hash != record["image_sha256"]:
            raise ValueError(f"Image changed since selection/export: {image_path}")
        if label_hash != item["label_sha256"] or label_hash != record["labels_sha256"]:
            raise ValueError(f"Labels changed since selection/export: {label_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        annotations = load_annotations(label_path, image.width, image.height)
        summaries = []
        for model, predictions in record["predictions"].items():
            counts = {}
            for target, boxes in annotations.items():
                matches, tp = match_predictions(predictions, boxes, box_iou)
                counts[target] = tp
                for match in matches:
                    if match["status"] == "TP":
                        all_refs.append({"sample_id": item["sample_id"], "image": item["image"],
                                         "model": model, "annotation_target": target, **match})
            prediction_count = sum(p["confidence"] >= 0.05 for p in predictions)
            baseline_rows.append({"sample_id": item["sample_id"], "image": item["image"], "model": model,
                                  "predictions_at_005": prediction_count, "fire_gt_count": len(annotations["fire"]),
                                  "fire_tp": counts["fire"], "smoke_gt_count": len(annotations["smoke"]),
                                  "smoke_tp": counts["smoke"]})
            summaries.append(f"{model}: fire matches={counts['fire']}, smoke matches={counts['smoke']}")
        draw = ImageDraw.Draw(image)
        for target, boxes in annotations.items():
            color = "lime" if target == "fire" else "cyan"
            for number, box in enumerate(boxes):
                draw.rectangle(box, outline=color, width=max(2, round(image.width / 400)))
                draw.text((max(0, box[0]), max(0, box[1])), f"{target} {number}", fill=color, stroke_width=1)
        preview = ImageOps.contain(image, (1280, 720))
        preview.save(output / f"{item['sample_id']}_annotations.png")
        local = index % 12
        if local == 0:
            sheet = Image.new("RGB", (1920, 1080), "white")
        left, top = (local % 4) * 480, (local // 4) * 360
        thumb = ImageOps.contain(image, (470, 260))
        sheet.paste(thumb, (left + (480 - thumb.width) // 2, top + 92))
        draw = ImageDraw.Draw(sheet)
        draw.text((left + 6, top + 5), item["sample_id"] + " | GT fire=green, smoke=cyan", fill="black")
        for line_number, text in enumerate(summaries):
            draw.text((left + 6, top + 24 + 16 * line_number), text, fill="black")
        draw.text((left + 6, top + 64), "Matches to supplied annotations, IoU 0.50, conf 0.05", fill="black")
        if local == 11 or index == len(manifest) - 1:
            sheet.save(output / f"annotation_contact_sheet_{index // 12 + 1}.png")
        # Preserve review choices. Add explicit cautions grounded in the uploaded contact sheet.
        updated = dict(item)
        updated["annotation_review"] = "pending_original_image_review"
        updated["provisional_event_group"] = item["source_group"]
        updated["audit_note"] = ""
        if image_path.name.startswith(("youtube-220_jpg.", "youtube-209_jpg.")):
            updated["provisional_event_group"] = "provisional_harbor_event"
            updated["audit_note"] = "Contact sheet suggests the same event as the other harbor image; fire annotation completeness needs review."
        if image_path.name.startswith("youtube-194_jpg."):
            updated["audit_note"] = "Visible flames in contact sheet but no supplied fire boxes; inspect original image and overlay."
        if image_path.name.startswith("youtube-84_jpg."):
            updated["audit_note"] = "Bright roof glow/possible flames; inspect original image before assigning flame visibility."
        review_rows.append(updated)
    write_csv(output / "validation_baseline.csv", baseline_rows)
    write_csv(output / "validation_review.csv", review_rows)
    if all_refs:
        write_csv(output / "matched_annotation_references.csv", all_refs)
    summary = {"completed": True, "manifest": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
               "baseline_export": str(baseline), "sample_size": len(manifest), "confidence": 0.05, "iou": 0.50,
               "inference": "Cached predictions only; no inference or training performed.",
               "review": "Potential annotation omissions and repeated events require visual/source review. Zero supplied fire boxes does not establish absence of visible flame.",
               "matching": "Fire and smoke overlaps are independent tasks; the same prediction can match in both. Do not sum these TP counts."}
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Reviewed {len(manifest)} fixed sample images using cached predictions.")
    print(f"Outputs saved to: {output}")
    print("Send annotation_contact_sheet_1.png and validation_baseline.csv for review.")


if __name__ == "__main__":
    main()
