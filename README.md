# Multimodal Deep Learning Framework for Hate, Toxicity, and Violence Detection in YouTube Shorts

A personal research project applying classical machine learning to multimodal content moderation.

A comparative study of ten classical machine learning algorithms on a **4,611-dimensional multi-modal feature set**
(BERT text embeddings + video + audio) for automated content moderation of YouTube Shorts. The project builds an
end-to-end pipeline — from data collection and Gemini-based auto-labelling, through BERT/VideoMAE/ResNet50/librosa
feature extraction, to a systematic benchmark of classical ML models across two label taxonomies (5-class and 4-class).

📄 Full write-up: [`docs/PROJECT\\\\\\\_REPORT.md`](docs/PROJECT_REPORT.md)



##⚖️ Credits & Legal Considerations



This project uses publicly available \*\*YouTube Shorts\*\* content, accessed via the YouTube Data API, solely for non-commercial academic research into content moderation; all original video, audio, and metadata remain the property of their respective creators and YouTube, LLC / Google LLC, and this repository contains \*\*no raw video, audio, or thumbnail files\*\* — only derived numerical features (BERT/VideoMAE/ResNet50/librosa embeddings), aggregate statistics, and trained model weights, which do not permit reconstruction of the source media. This project is \*\*not affiliated with, endorsed by, or sponsored by YouTube or Google LLC\*\*, and nothing here constitutes legal advice.



📊 Every result explained in plain language: [`docs/RESULTS\\\\\\\_EXPLAINED.md`](docs/RESULTS_EXPLAINED.md)

\---

## 🔑 Key Results

|Task|Best (single split)|Best (5-fold CV)|
|-|-|-|
|**5-class** (Explicit, Violence, Hateful, Normal, Toxic)|Random Forest — 83.46% acc|Logistic Regression — 74.13% acc / 74.09% F1|
|**4-class** (Hateful merged into Toxic)|Neural Network — 89.47% acc|**XGBoost — 84.72% acc / 84.46% F1**|

* Merging the ambiguous **Hateful** class into **Toxic** (rather than dropping it) improved cross-validated accuracy for *every* model, by **9–19 percentage points**.
* Modality contribution (single-modality, 5-class, Logistic Regression): **Video (67.8%) > Text/BERT (59.3%) > Audio (58.1%)**.
* Combining all modalities (comments + video + audio) gave the strongest overall result: **85.78% accuracy / 85.68% F1** (Logistic Regression, 4-class).
* Recommended deployment configuration: **Logistic Regression** on **BERT (comments) + Video + Audio**, 4-class taxonomy — with the interpretability and low compute cost of a linear model.

See [`docs/RESULTS\\\\\\\_EXPLAINED.md`](docs/RESULTS_EXPLAINED.md) for the full breakdown, including a confusion matrix and per-class error analysis.

## 🧩 Pipeline Overview

```
YouTube Shorts URLs (seed CSV, 440 videos)
        │
        ▼
 yt-dlp download  ──────────────►  labelling (5 classes)
        │                                   │
        ▼                                   ▼
 ┌─────────────┐   ┌───────────────┐   ┌────────────┐   ┌──────────────────┐
 │ Video frames│   │ Audio (mp4→mp3)│  │  Title      │   │ Comments (YouTube│
 │             │   │                │   │             │   │ Data API v3)     │
 └──────┬──────┘   └───────┬────────┘   └──────┬──────┘   └────────┬─────────┘
        │                  │                    │                   │
 VideoMAE (768) +   librosa: MFCC, chroma,   BERT \\\\\\\[CLS]         BERT \\\\\\\[CLS]
 ResNet50 (2048) +  spectral\\\\\\\*, tonnetz,      embedding (768)    avg-pooled
 Optical Flow (120) tempo, RMS (139)                            embedding (768)
        │                  │                    │                   │
        └──────────────────┴────────────────────┴───────────────────┘
                                   │
                     Combined feature vector (4,611-dim)
                                   │
                                   ▼
              10 classical ML models × {5-class, 4-class}
      (Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest,
       AdaBoost, Gradient Boosting, XGBoost, SVM, Neural Network/MLP)
                                   │
                                   ▼
                    Evaluation: Accuracy / Precision / Recall / F1
                 (single 80/20 stratified split + 5-fold stratified CV)
```

## 📊 Dataset

* **661** labelled YouTube Shorts samples across 5 classes (Explicit, Normal, Hateful, Toxic, Violence — roughly balanced, \~130 per class).
* Labels were generated automatically using the **Gemini 2.5 Flash** model (multimodal: visual frames + audio cues + text), then used as ground truth.
* Comments extracted via **YouTube Data API v3** (up to 100 top-level comments/video; 18,537 comments across 324 videos).

**Final feature vector (4,614 dims):**

|Component|Dims|Source|
|-|-|-|
|Title embedding|769|BERT (`bert-base-uncased`), \[CLS] token|
|Comment embedding|768|BERT, \[CLS] token, averaged over comments|
|Video features|2,920|VideoMAE (768) + ResNet50 (2048) + Optical Flow (120), from 16 sampled frames|
|Audio features|139|librosa: MFCC, spectral centroid/rolloff/bandwidth, ZCR, tempo, chroma, RMS energy, spectral contrast, tonnetz|

> Raw videos/audio are \\\\\\\*\\\\\\\*not\\\\\\\*\\\\\\\* included in this repository (see \\\\\\\[Data Availability](#-data-availability)).

## 📁 Repository Structure

```
.
├── notebooks/
│   ├── 0-url\\\\\\\_to\\\\\\\_BERT\\\\\\\_video\\\\\\\_audio\\\\\\\_features\\\\\\\_combined\\\\\\\_R2.ipynb   # End-to-end feature extraction pipeline
│   ├── 1-ML\\\\\\\_classification\\\\\\\_BERT.ipynb                         # BERT-only baseline
│   ├── 2-ML\\\\\\\_classification\\\\\\\_video\\\\\\\_features.ipynb               # Video-only baseline
│   ├── 3-ML\\\\\\\_classification\\\\\\\_audio\\\\\\\_features.ipynb                # Audio-only baseline
│   ├── 4-ML\\\\\\\_classification\\\\\\\_BERT\\\\\\\_audio\\\\\\\_video\\\\\\\_features.ipynb    # BERT + audio + video, 5-class
│   ├── 5-ML\\\\\\\_classification\\\\\\\_BERT\\\\\\\_video\\\\\\\_features.ipynb          # BERT + video
│   ├── 6-ML\\\\\\\_classification\\\\\\\_BERT\\\\\\\_audio\\\\\\\_video\\\\\\\_features\\\\\\\_4Category.ipynb  # BERT + audio + video, 4-class
│   ├── 7-ML\\\\\\\_classification\\\\\\\_BERT\\\\\\\_audio\\\\\\\_video\\\\\\\_title.ipynb       # + title, 5-class
│   ├── 8-ML\\\\\\\_classification\\\\\\\_BERT\\\\\\\_audio\\\\\\\_video\\\\\\\_title\\\\\\\_4labels.ipynb # + title, 4-class (best config)
│   └── 8-title\\\\\\\_to\\\\\\\_avg\\\\\\\_BERT.ipynb                              # Title → BERT embedding utility
├── docs/
│   ├── PROJECT\\\\\\\_REPORT.md                                       # Full write-up: background, pipeline, methodology
│   └── RESULTS\\\\\\\_EXPLAINED.md                                    # Every results table explained in plain language
├── requirements.txt
├── LICENSE
└── README.md
```

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/youtube-shorts-content-moderation.git
cd youtube-shorts-content-moderation
python -m venv venv \\\\\\\&\\\\\\\& source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
jupyter notebook notebooks/
```

Notebook `0-url\\\\\\\_to\\\\\\\_BERT\\\\\\\_video\\\\\\\_audio\\\\\\\_features\\\\\\\_combined\\\\\\\_R2.ipynb` requires:

* A **YouTube Data API v3 key** for comment extraction.
* `ffmpeg` installed and on your `PATH` for audio extraction.

Set credentials as environment variables (e.g. `GEMINI\\\\\\\_API\\\\\\\_KEY`, `YOUTUBE\\\\\\\_API\\\\\\\_KEY`) before running — do **not** hard-code keys in the notebooks.

## 🧠 Models Evaluated

Logistic Regression · Naive Bayes · K-Nearest Neighbors · Decision Tree · Random Forest ·
AdaBoost · Gradient Boosting · XGBoost · Support Vector Machine (RBF) · Neural Network (MLP)

Evaluated with a stratified 80/20 train-test split **and** 5-fold stratified cross-validation, reporting weighted
Accuracy, Precision, Recall, and F1-score.

## 📈 Limitations \& Future Work

* Small dataset (661 samples) limits generalization, especially for high-capacity models.
* No hyperparameter tuning (GridSearch/RandomSearch) was performed — a stated direction for future work.
* No dimensionality reduction (PCA) applied to the 4,614-dim space.
* Class imbalance mitigation (SMOTE, cost-sensitive learning) and explainability (SHAP/LIME) are left as future work.
* See [`docs/PROJECT\\\\\\\_REPORT.md`](docs/PROJECT_REPORT.md) for the complete discussion and roadmap.

## 📚 References

Key references include Devlin et al. (2019, BERT), Chen \& Guestrin (2016, XGBoost), and Pedregosa et al. (2011,
scikit-learn). Full reference list in [`docs/PROJECT\\\\\\\_REPORT.md`](docs/PROJECT_REPORT.md).

## 🗂 Data Availability

Due to size and YouTube content-redistribution restrictions, raw videos, audio files, and extracted feature CSVs are
not published in this repository. The notebooks document the exact extraction pipeline so results can be reproduced
on a newly collected dataset using the same seed-URL → download → label → feature-extract workflow.

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the file for details. Please credit this repository
if you build on this work.

