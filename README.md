# Data-Centric Cohort Refinement for Clinical Risk Prediction

This repository contains the full implementation of our paper:

**"A Data-Centric and Interpretable Framework for Clinical Risk Prediction via Cohort Refinement"**

We present a modular framework that improves predictive modeling by refining patient cohorts using odds ratio filtering, Isolation Forest anomaly detection, and Shapley-based sample attribution.

---

## 📂 Repository Structure

```
code/
├── main_pipeline.py      # Main pipeline: cohort refinement, training, SHAP filter/weight experiments
├── pca_analysis.py       # PCA + KMeans visualization for each refinement strategy
datasets/
├── labevents.csv         # Raw lab measurements from MIMIC-IV
├── mimc_organize.csv     # Admission records with subject_id and admittime
supplement/
├── data-centric-cohort-refinement_supplemental.pdf  # Full evaluation tables (AUC, F1, Recall...)
_results/
├── versionXX_*.csv/.pkl/.tex  # Output results (optional)
```

---

## Dataset: MIMIC-IV

This project uses the publicly available [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) database from PhysioNet.

> ⚠Due to data use agreements, we do not include raw MIMIC-IV files in this repository.  
> Please obtain access via PhysioNet and prepare the following:
- `labevents.csv`: laboratory events (including creatinine, ItemID 50912)
- `mimc_organize.csv`: organized admissions file with `subject_id` and `admittime`

---

## ⚙Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `pandas`, `numpy`, `scikit-learn`
- `keras` (TensorFlow backend)
- `xgboost`
- `matplotlib`, `seaborn`, `tqdm`

---

## How to Run

**Main pipeline (training + SHAP experiments):**
```bash
python code/main_pipeline.py
```

**PCA clustering visualization:**
```bash
python code/pca_analysis.py
```

---

## Supplementary Materials

All evaluation metrics (Accuracy, F1, Recall, etc.) and per-version scores are provided in:

[`supplement/data-centric-cohort-refinement_supplemental.pdf`](./supplement/data-centric-cohort-refinement_supplemental.pdf)

---

## Figures

We include all main and supplementary figures used in the paper in the `figures/` folder for reference.

These figures are referenced in the main paper and supplement.

---

## 📜 License

This code is released for academic use only under a custom research license.  
Please cite the paper if used and contact the authors for commercial use.
