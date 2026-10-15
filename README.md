---
license: cc-by-nc-sa-4.0
extra_gated_fields:
  Name: text
  Email: text
  Country: country
  Organization or Affiliation: text
  What do you intend to use the model for?:
    type: select
    options:
      - Research
      - Personal use
      - Creative Professional
      - Startup
      - Enterprise
---
# Dataset Card: CT-ScanGaze

**Dataset Name**: `phamtrongthang/CT-ScanGaze`  
**Repository**: [UARK‑AICV/CTScanGaze](https://github.com/UARK-AICV/CTScanGaze)  
**License**: CC BY-SA 4.0

---

## 1. Dataset Summary

CT-ScanGaze is the first publicly available dataset that captures **3D eye-gaze trajectories of expert radiologists** while interpreting volumetric CT scans. It includes:

- 1045 abdominal and chest CT scans (`.nii.gz` format)  
- Expert radiologist 3D gaze sequences (x, y, z with timestamp and duration)  
- Transcribed radiology reports

This dataset supports research in gaze modeling, scanpath prediction, explainable AI, and multi-modal learning in medical imaging.

> 🏅 This work was acceppted as a **highlight paper at ICCV 2025**.

---

## 2. Dataset Structure

| Attribute            | Description |
|---------------------|-------------|
| **Total Volumes**   | 1045 CT scans |
| **Modality**        | CT (`.nii.gz`) |
| **Gaze Data**       | 3D coordinates + fixation duration |
| **Reports**         | Transcribed from audio recordings |
| **Radiologists**    | Two experts (10+ years of experience) |

---

## 3. Intended Uses

- 3D scanpath prediction  
- Gaze-informed diagnosis modeling  
- Medical report generation  
- Visual search behavior analysis  
- Explainability in medical AI

---

## 4. Tasks and Benchmarks

**Primary Task**: 3D Scanpath Prediction  
- Baseline: **CT‑Searcher** (transformer-based model)  
- Evaluation Metrics:
  - **Scanpath**: ScanMatch, MultiMatch, SED  
  - **Saliency**: CC, KL Divergence, NSS

Other potential tasks:
- Report–gaze alignment  
- Gaze-informed classification  
- Multi-modal fusion (gaze + imaging + text)

## 5. One JSON Trajectory per CT

`Data collection-MD1` and `Data collection-MD2` contain Tobii TSV exports at the
recording/session level. They do not contain one row or one file per CT volume.
The case-level 3D trajectories are indexed by `mds_report.json`; each `name`
matches exactly one `cts/<name>.nii.gz` file.

To split the manifest into one JSON file per volume, run:

```bash
python3 scripts/split_visual_trajectories.py
```

This writes `trajectories/<name>.json`. Each file contains `image` and a
`trajectories` array; every item has `x/y/z`, `duration`, and the source fields
under `metadata`. A single-trajectory file also exposes that item as
`trajectory` for convenience. Some volumes have multiple subjects/groups, so
all paths are retained instead of silently selecting one. Coordinates are
preserved as exported voxel-space values; boundary values such as `512` are not
silently clipped. Apply the boundary policy required by the downstream model
after reading the NIfTI shape. `duration` is preserved as the exported
per-point dwell time.

---

## 6. Data Availability

The processed dataset (`.nii.gz`, 3D gaze, and reports) is publicly available via Hugging Face.  
**Raw data**, including original **DICOM files** and **video recordings of the interpretation sessions**, are provided as "raw_data.zip".

👉 To request access to the raw data, please contact: **tp030@uark.edu**

---

## 6. Citation

Please cite this dataset using the following BibTeX entry:

```bibtex
@article{pham2025ct,
  title={CT-ScanGaze: A Dataset and Baselines for 3D Volumetric Scanpath Modeling},
  author={Pham, Trong-Thang and Awasthi, Akash and Khan, Saba and Marti, Esteban Duran and Nguyen, Tien-Phat and Vo, Khoa and Tran, Minh and Nguyen, Ngoc Son and Van, Cuong Tran and Ikebe, Yuki and others},
  journal={arXiv preprint arXiv:2507.12591},
  year={2025}
}
```
