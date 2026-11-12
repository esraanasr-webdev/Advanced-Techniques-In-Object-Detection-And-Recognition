# 📜 Project Changelog & Contribution Matrix

> **Project:** Cognitive Fire Defense: Zero-Shot Smoke-to-Fire Generalization in Object Detection  
> **Course:** AIN7601 — Advanced Techniques in Object Detection and Recognition  
> **Authors:** Esraa Nasr & Ahmed Ayman  
> **Institution:** Arab Academy for Science, Technology and Maritime Transport (AASTMT), College of Artificial Intelligence  

---

## 👥 Authors & Contribution Matrix

This project was built collaboratively through distinct modular division of architectural responsibilities, data pipelines, and research deliverables.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            COGNITIVE FIRE DEFENSE                           │
├──────────────────────────────────────┬──────────────────────────────────────┤
│             ESRAA NASR               │             AHMED AYMAN              │
│       (CNN & Edge Paradigms)         │     (Transformers & Data Systems)    │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Lead: YOLO11n Single-Stage CNN     │ • Lead: RT-DETR Hybrid Transformer   │
│ • Lead: Faster R-CNN Two-Stage RPN   │ • Lead: Deformable DETR Sparse Model │
│ • Custom Augmentation Design (HSV,   │ • 80/20 Leak-Free Video-Clip Split   │
│   Mosaic, Scale Jitter)              │ • YOLO-to-COCO Data Pipeline Engine  │
│ • Domain-Specific Anchor Clustering  │ • Focal Loss Rebalancing (alpha=0.95)│
│ • Strict-IoU Zero-Shot Evaluation    │ • Relaxed Early-Warning Metric Design│
│ • Qualitative Fire Probing (CNNs)    │ • Fast-Track Ablation Runner Suite   │
│ • IEEE Paper Word Draft & Baseline   │ • Full IEEE LaTeX Master & Overleaf  │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### Detailed Contribution Breakdown

#### 👩‍💻 **Esraa Nasr (CNN & Anchor-Based Systems)**
1. **YOLO11n (Single-Stage CNN Edge Baseline):**
   * Designed baseline and custom augmentation training pipelines in Ultralytics.
   * Investigated impact of mosaic reduction, scale jitter, and HSV shifts on smoke edge detection.
   * Conducted 100-epoch convergence analysis and validation evaluations.
2. **Faster R-CNN + MobileNetV3-FPN (Two-Stage RPN):**
   * Implemented custom PyTorch training loop with MobileNetV3-Large FPN backbone.
   * Extracted K-means ($k=5$) domain-specific anchor priors from Dataset B bounding boxes to handle 95.7% large plume bias.
   * Analyzed RPN proposal generation behavior under extreme aspect ratio shifts.
3. **CNN Zero-Shot Transfer Probing:**
   * Evaluated both YOLO11n and Faster R-CNN on Dataset A (unseen fire).
   * Proved empirically that standard CNN feature maps fail to transfer zero-shot across the smoke-to-fire domain shift under strict box-level IoU matching.
4. **Research Documentation:**
   * Authored original experimental logs, CNN analysis reports, and initial paper draft deliverables.

---

#### 👨‍💻 **Ahmed Ayman (Transformer Architectures, Data Systems & Integration)**
1. **RT-DETR (Real-Time Hybrid Transformer):**
   * Configured, trained, and tuned RT-DETR with hybrid CNN-Transformer encoder and NMS-free decoder.
   * Benchmarked inference latency and zero-shot attention map transfers.
2. **Deformable DETR (Sparse Multi-Scale Attention):**
   * Led the transformer architectural progression: diagnosed why Vanilla DETR (quadratic memory cost) and DINO (contrastive query collapse on single-class data) failed.
   * Adopted `SenseTime/deformable-detr` and engineered the **critical focal loss gradient fix (`focal_alpha = 0.95`)** to prevent "dead head" background gradient suppression (~299:1 imbalance).
   * Implemented zero-overhead in-memory RAM caching to resolve Windows multiprocessing deadlock issues.
3. **Data Splitting & Pipeline Infrastructure:**
   * Designed the **clip-level leak-free data splitting algorithm** (80/20 train/val) preventing temporal video leakage.
   * Built `yolo_to_coco.py` data converter and unified COCO annotation structures.
4. **Relaxed Early-Warning Evaluation Framework:**
   * Formulated the relaxed localization metric ($\text{IoU} \ge 0.10$), demonstrating that Transformers achieve ~58.82% fire anomaly sensitivity without ever seeing fire training labels.
5. **Ablations, Scripts & Paper Finalization:**
   * Developed `ablation_runner.py` for rapid constrained ablation testing.
   * Built and maintained the IEEE LaTeX manuscript (`cognitive_fire_defense.tex`), Overleaf integration, and strict passive-voice / de-AI academic compliance.

---

## 📊 Comprehensive Experimental Results

### 1. Within-Domain Smoke Detection (Dataset B — 80/20 Clip Split)

| Model | Architecture Paradigm | Framework | Epochs | Precision | Recall | F1 Score | mAP@50 | mAP@50-95 | Lead Owner |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO11n (Baseline)** | One-Stage CNN | Ultralytics | 40 | 89.51% | 86.86% | 88.15% | 94.01% | 62.01% | Esraa Nasr |
| **YOLO11n (Custom Aug)** | One-Stage CNN | Ultralytics | 70 | 92.70% | 92.81% | 92.75% | **96.51%** | 60.27% | Esraa Nasr |
| **Faster R-CNN (MobileNetV3)** | Two-Stage RPN | PyTorch/torchvision | 5 | **95.27%** | **94.47%** | **94.87%** | 94.18% | **63.24%** | Esraa Nasr |
| **Deformable DETR** | Sparse Attention Transformer | HuggingFace | 50 | — | — | — | 88.61% | 40.75% | Ahmed Ayman |

---

### 2. Zero-Shot Fire Transfer (Dataset A — Strict Matching $\text{IoU} \ge 0.50$)

| Model | Conf Threshold | Predictions | True Positives | False Positives | False Negatives | Precision | Recall | F1 Score | Image Det Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n (Custom)** | 0.05 | 1,628 | 6 | 1,622 | 989 | 0.37% | 0.60% | 0.46% | 1.31% (6/459) |
| **Faster R-CNN** | 0.05 | 2,856 | 51 | 2,805 | 944 | 1.79% | 5.13% | 2.65% | 11.11% (51/459) |

---

### 3. Zero-Shot Fire Transfer Under Relaxed Localization ($\text{IoU} \ge 0.10$)

| Model | Conf Threshold | Predictions | True Positives | False Positives | False Negatives | Precision | Recall | F1 Score | Image Det Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---:cross | :---: | :---: | :---: |
| **RT-DETR** | 0.50 | 482 | 62 | 420 | 933 | 12.86% | 6.23% | 8.40% | 13.07% (60/459) |
| **RT-DETR** | 0.10 | 3,600 | 131 | 3,469 | 864 | 3.64% | 13.17% | 5.70% | 25.27% (116/459) |
| **RT-DETR** | 0.05 | 12,942 | 188 | 12,754 | 807 | 1.45% | 18.89% | 2.70% | 33.33% (153/459) |
| **Deformable DETR** | 0.50 | 765 | 57 | 708 | 938 | 7.45% | 5.73% | 6.48% | 12.42% (57/459) |
| **Deformable DETR** | 0.10 | 19,656 | 265 | 19,391 | 730 | 1.35% | 26.63% | 2.57% | 42.92% (197/459) |
| **Deformable DETR** | 0.05 | 46,983 | 431 | 46,552 | 564 | 0.92% | **43.32%** | 1.80% | **58.82% (270/459)** |

---

## 🔄 Version & Development Changelog

### **v1.4.0 (Final Publication & Integration Phase) — [Current]**
* **Deformable DETR Module Finalized:** Created dedicated sub-module documentation (`dataset-b/model_training/dino/README.md`) detailing why Vanilla DETR & DINO were superseded by Deformable DETR.
* **Fast-Track Ablation Runner Suite:** Implemented `ablation_runner.py` for rapid constrained ablation testing (sub-sampling + reduced epochs) as per conference guidelines.
* **Master IEEE Conference LaTeX Finalization:** 
  * Fixed document class from `[journal]` to `[conference]`.
  * Embedded multi-column figures: Study Design (`study_design.png`), Saber Architecture (`fig3_saber.png`), and Architectural Overview of the Four Detectors (`fig2_architectures.jpeg`).
  * Conducted full de-AI linguistic sweep and strict passive voice enforcement.
* **Repository Cleanup & Checklist Audit:** Passed AG Kit automated test and security verification suite with 100% pass rate.

---

### **v1.3.0 (Transformer & Zero-Shot Breakthrough Phase)**
* **Deformable DETR Convergence Breakthrough:** Overrode `focal_alpha=0.95` to solve positive gradient suppression in single-class object detection.
* **Relaxed Localization Protocol:** Introduced IoU $\ge 0.10$ early-warning metric on Dataset A, proving Transformers achieve ~60% anomaly detection on unseen fire.
* **RT-DETR Model Pipeline:** Integrated real-time transformer architecture without NMS post-processing.
* **COCO Data Standardization:** Implemented `yolo_to_coco.py` to support multi-framework benchmarking.

---

### **v1.2.0 (CNN Modeling & Anchor Optimization Phase)**
* **YOLO11n Augmentation Tuning:** Analyzed mosaic reduction (0.4), HSV jitter, and scale augmentation effects on boreal smoke.
* **Faster R-CNN Domain Anchors:** Extracted $k=5$ K-means bounding box cluster priors for RPN injection.
* **Initial Zero-Shot Evaluation (Dataset A):** Conducted box-level matching evaluation and established baseline CNN zero-shot failure modes.

---

### **v1.1.0 (Data Engineering & Preprocessing Phase)**
* **Temporal Leakage Elimination:** Implemented video-clip level splitting (80/20 train/val) preventing duplicate frame memorization.
* **Data Integrity Audit:** Audited 4,954 images, identifying 4,139 annotation flags without ground-truth corruption.
* **Dataset A (Fire Probe) Integration:** Ingested 637 evaluation images (459 with fire, 995 fire boxes).

---

### **v1.0.0 (Project Conception & Architecture Setup Phase)**
* Formulated core research hypothesis: *"Can an object detector trained exclusively on smoke recognize fire zero-shot?"*
* Initialized codebase structure, environment dependencies, and four-paradigm architectural roadmap.
