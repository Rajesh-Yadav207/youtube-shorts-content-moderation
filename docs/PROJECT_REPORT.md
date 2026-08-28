# Project Report: A Multimodal Deep Learning Framework for Hate, Toxicity, and Violence Detection in YouTube Shorts

*A personal research project applying classical machine learning to multimodal content moderation.*

\---

## Abstract

The rapid proliferation of user-generated video content on platforms such as YouTube presents significant
challenges for automated content moderation. Harmful content often manifests across multiple modalities —
textual titles, audio tracks, visual frames, and user comments — necessitating multimodal approaches for
effective detection.

This project presents a comprehensive comparative study of ten classical machine learning algorithms applied
to a multimodal feature set for YouTube content classification. The feature set comprises 4,614 features
derived from BERT embeddings of titles and comments, alongside audio and video features. Two classification
taxonomies are evaluated: a 5-class scheme (Explicit, Violence, Hateful, Normal, Toxic) and a 4-class scheme
(excluding Hateful).

Experimental results demonstrate that multimodal features enable effective classification, with the 4-class
scheme substantially outperforming the 5-class scheme across all models. In cross-validation, **XGBoost**
achieves the highest performance on the 4-class task with **84.72% accuracy** and **84.46% weighted F1-score**,
while a **Neural Network** attains the peak single-split accuracy of **89.47%**. The findings highlight the
critical impact of label taxonomy design and the viability of classical machine learning models for
multimodal content moderation tasks.

For a plain-language breakdown of every result table, see [`RESULTS\\\_EXPLAINED.md`](RESULTS_EXPLAINED.md).

\---

## 1\. Introduction

### 1.1 Background

The digital content ecosystem has expanded exponentially over the past decade, with video-sharing platforms
like YouTube becoming primary channels for information dissemination, entertainment, and social interaction.
YouTube hosts billions of videos and serves over two billion monthly active users, generating a volume of
content that far exceeds the capacity of manual human moderation.

Content moderation — the set of practices and technologies used to identify, evaluate, and manage
user-generated content that violates platform policies — has become a critical operational requirement for
digital platforms. Effective moderation is essential for protecting users from harmful material, maintaining
platform integrity, and complying with increasingly stringent regulatory requirements.

### 1.2 Problem Statement

A fundamental challenge in automated content moderation is that harmful content rarely manifests through a
single information channel. A video may contain abusive language in its audio track, explicit imagery in its
visual frames, offensive content in its title, and toxic discourse in its comment section. Traditional
moderation systems that rely exclusively on textual analysis are inherently limited in their ability to detect
harmful content conveyed through non-textual means.

This project addresses the problem of automated YouTube content classification by developing and evaluating
machine learning models that leverage multimodal features extracted from video titles (via BERT embeddings),
audio signals, visual content, and comment metadata.

### 1.3 Objectives

* Extract and integrate multimodal features from YouTube video data.
* Evaluate the performance of ten classical machine learning algorithms on the multimodal feature set.
* Compare two classification taxonomies: a 5-class scheme and a 4-class scheme.
* Identify the most effective models for multimodal content classification.
* Provide empirical insights for the development of practical content moderation systems.

### 1.4 Scope

This project focuses on the classification of YouTube videos into predefined harmful-content categories using
classical machine learning algorithms. The scope is limited to:

* A dataset of 661 labelled YouTube video samples.
* Features extracted offline and provided as numeric vectors.
* Ten widely used machine learning classifiers.
* Two label taxonomies (5-class and 4-class).
* Deep learning architectures and end-to-end feature learning are outside the scope of this work.

\---

## 2\. Related Concepts

### 2.1 Multimodal Learning for Content Moderation

Multimodal learning refers to the integration of information from multiple sensory modalities to improve
model performance. In the context of content moderation, the primary modalities include text (titles,
descriptions, comments), audio (speech and background sounds), and visual content (video frames and
thumbnails). Combining these signals consistently outperforms unimodal methods in detecting harmful content.

### 2.2 Transformer-Based Text Representation

BERT (Bidirectional Encoder Representations from Transformers) achieves strong performance on text
classification benchmarks, including hate-speech and toxicity detection. In this project, BERT embeddings
extracted from video titles and comments serve as the primary textual features.

### 2.3 Classical Machine Learning for Classification

While deep learning approaches have gained prominence, classical machine learning algorithms remain widely
used in content moderation due to their interpretability, training efficiency, and ease of deployment.
Algorithms such as Logistic Regression, Random Forest, SVM, and Gradient Boosting have been extensively
applied to text and multimodal classification tasks; ensemble methods often achieve strong performance by
combining multiple weak learners.

### 2.4 Motivation for This Project

Relatively few projects systematically compare classical machine learning algorithms on multimodal YouTube
content features, and the impact of label-taxonomy design — specifically, the number and definition of harmful
categories — on classification performance is often under-examined. This project addresses that gap with an
empirical evaluation across ten algorithms and two classification schemes.

\---

## 3\. Data Collection and Labelling

### 3.1 Overview

A robust supervised machine learning model is fundamentally dependent on the quality and representativeness of
its training data. For this project, a custom dataset of YouTube Shorts videos was curated, spanning five
distinct content categories: **Explicit, Hateful, Normal, Toxic, and Violence**.

### 3.2 Data Source: YouTube Shorts

YouTube Shorts was selected as the sole data source due to its contemporary relevance and multimodal
richness. Shorts are short-form vertical videos, typically under 60 seconds, and combine visual frames, audio
tracks, textual titles, and user comments — making them ideal candidates for multimodal classification.

### 3.3 Data Collection Pipeline

The data collection process was fully automated using a Python script executed in a Jupyter notebook
(`notebooks/0-url\\\_to\\\_BERT\\\_video\\\_audio\\\_features\\\_combined\\\_R2.ipynb`). The pipeline ingests a seed dataset,
downloads the corresponding videos, and organises them into a structured directory hierarchy.

**Tools and dependencies:**

|Library|Purpose|
|-|-|
|`yt-dlp`|Downloading YouTube videos with format selection and cookie authentication|
|`google-generativeai`|Interfacing with the Gemini API for automated video labelling|
|`pandas`|Managing and iterating over the input CSV|
|`os` / `pathlib`|File-system operations and path management|

**Seed dataset:** A seed CSV file contains `url`, `title`, and `label` columns. A total of **440 distinct
URLs** were included, with an approximately uniform distribution across the five categories.

**Automated download workflow (per iteration):**

1. **Directory verification** — a subfolder is created per label to group videos by category.
2. **Title sanitisation** — video titles often contain characters (emojis, slashes, quotes) illegal in file
paths; a sanitisation function strips these.
3. **Video download** — using `yt-dlp`, only the video stream (no audio) is downloaded, since audio is
extracted separately.
4. **Rate-limiting delay** — a randomised 10–18 second delay between downloads avoids triggering YouTube's
anti-bot mechanisms.

### 3.4 Dataset Statistics

A total of **580 videos** were processed for download, and the final labelled feature dataset contains
**661 samples**, roughly balanced across classes:

|Label|Count|
|-|-|
|Explicit|129|
|Normal|137|
|Hateful|133|
|Toxic|133|
|Violence|129|
|**Total**|**661**|

### 3.5 Challenges and Mitigations

|Challenge|Mitigation|
|-|-|
|**Video unavailability** — some seed URLs were removed by uploaders or taken down for policy violations|`try/except` blocks log the failure and continue the pipeline|
|**Rate limiting / HTTP 403** — YouTube occasionally blocks automated access|Randomised sleep intervals simulate human browsing behaviour|
|**Duplicate entries** — the seed CSV contained duplicate URLs with differing query parameters|Filename sanitisation + `yt-dlp` overwrite behaviour prevent storage duplication|

\---

## 4\. Comments → BERT Embeddings

### 4.1 Comment Extraction via YouTube API

Comments are extracted using the **YouTube Data API v3** (`notebooks` comment-extraction step): for each
video ID (parsed via regex from Shorts or watch URLs), the script calls `commentThreads().list()` requesting
up to 100 top-level comments per video, with error handling for comments-disabled videos, missing videos, and
quota limits, and a 0.3-second delay between requests. For a dataset of 324 videos this extracted **18,537
comment rows**.

### 4.2 BERT Embedding Generation

Using the pre-trained `bert-base-uncased` model from Hugging Face Transformers:

1. Each comment is tokenised (max sequence length 128, padding/truncation).
2. The `\\\[CLS]` token embedding is extracted from the last hidden layer as an aggregate representation.
3. **Average pooling** across all comments per video produces one 768-dimensional vector representing the
entire comment section.

### 4.3 Feature Vector Structure

|Component|Count|Description|
|-|-|-|
|`youtube\\\_link`|1|Original video URL|
|`Label`|1|Category label|
|`feature\\\_0` … `feature\\\_767`|768|BERT average embeddings|

\---

## 5\. Video Feature Extraction

Implemented in `notebooks/0-url\\\_to\\\_BERT\\\_video\\\_audio\\\_features\\\_combined\\\_R2.ipynb`, using three complementary
approaches over 16 uniformly sampled frames per video:

1. **VideoMAE (768 dims)** — `MCG-NJU/videomae-base`, a video transformer pre-trained on video data; the
mean-pooled last-hidden-state output captures spatio-temporal patterns beyond static frame analysis.
2. **ResNet50 (2048 dims)** — pre-trained on ImageNet (classification head removed), frames resized to
224×224, average-pooled across the 16 frames to capture objects, scenes, and visual semantics.
3. **Optical Flow (120 dims)** — Farnebäck algorithm (`cv2.calcOpticalFlowFarneback`) over 30 sampled frame
pairs; mean and standard deviation of motion magnitude and angle capture sudden movement and action
intensity.

**Combined video vector:** 768 + 2048 + 120 = **2,936 dimensions**.

\---

## 6\. Audio Feature Extraction

### 6.1 MP4 → MP3 Conversion

Audio tracks are isolated from video files using `ffmpeg`.

### 6.2 Acoustic Feature Extraction (librosa)

|Function|Dims|Description|
|-|-|-|
|MFCC|78|13 MFCCs × 6 (mean/std of raw, Δ, Δ²)|
|Spectral Centroid|2|Brightness of sound|
|Spectral Rolloff|2|Sharpness of sound|
|Spectral Bandwidth|2|Frequency spread|
|Zero Crossing Rate|2|Speech/music detection|
|Tempo|1|Beats per minute|
|Chroma|24|Pitch class distribution (12 classes × mean/std)|
|RMS Energy|2|Loudness|
|Spectral Contrast|14|Timbre/texture across 7 frequency bands|
|Tonnetz|12|Tonal/harmonic content|
|**Total**|**139**||

### 6.3 Output

Saved as `audio\\\_features.csv`: `Filename`, `feature\\\_0`…`feature\\\_138` (139 dims), `class\\\_label`.

\---

## 7\. Summary of Feature Extraction

|Feature Category|Dimensionality|Data Type|
|-|-|-|
|Title (BERT)|768|Float|
|Comments (BERT)|768|Float|
|Video (VideoMAE + ResNet50 + Optical Flow)|2,936|Float|
|Audio (librosa)|139|Float|
|**Combined vector**|**4611**|Float|

\*Slight variation across notebooks (4611) depending on whether the URL/title metadata columns are
---

## 8\. Methodology

### 8.1 Dataset

661 YouTube video samples, each annotated with one of five harmful-content labels, represented by a
multimodal feature set of 4,611 numeric features per sample (raw `url` / `title` text columns excluded from
the feature matrix since they were already encoded into embeddings).

### 8.2 Label Taxonomies

**5-class scheme:**

|Label|Definition|
|-|-|
|Explicit|Sexually explicit or profane content|
|Violence|Content depicting or promoting violence|
|Hateful|Content expressing hatred toward individuals or groups|
|Normal|Non-harmful content|
|Toxic|Content containing toxic or abusive language|

**4-class scheme:** the Hateful class is merged with Toxic, leaving Explicit, Normal, Toxic, and Violence.

### 8.3 Feature Pre-processing

Feature distributions showed significant scale variation (max |mean| ≈ 4,010; max |std − 1| ≈ 1,042), so
`StandardScaler` (`X\\\_scaled = (X − μ) / σ`) was applied to scale-sensitive models: **Logistic Regression, KNN,
SVM, Neural Network**. Tree-based models (**Decision Tree, Random Forest, AdaBoost, Gradient Boosting,
XGBoost**) were trained on unscaled features, since they are invariant to monotonic transformations.

### 8.4 Experimental Design

* **Train/test split:** stratified 80/20 split (`random\\\_state=44`) to preserve class balance.
* **Cross-validation:** 5-fold `StratifiedKFold` (`shuffle=True, random\\\_state=42`) for robust performance
estimates.

### 8.5 Evaluation Metrics

Weighted Accuracy, Precision, Recall, and F1-score (to account for class imbalance).

### 8.6 Models Evaluated

|Model|Key hyperparameters|
|-|-|
|Logistic Regression|`max\\\_iter=1000, solver='saga'`|
|Naive Bayes|`GaussianNB()`|
|K-Nearest Neighbors|`n\\\_neighbors=5`|
|Decision Tree|`random\\\_state=42`|
|Random Forest|`n\\\_estimators=100, random\\\_state=42`|
|AdaBoost|`n\\\_estimators=100, random\\\_state=42`|
|Gradient Boosting|`random\\\_state=42`|
|XGBoost|`random\\\_state=42, eval\\\_metric='mlogloss'`|
|Support Vector Machine|`kernel='rbf', probability=True`|
|Neural Network (MLP)|`hidden\\\_layer\\\_sizes=(128,64,32), max\\\_iter=300`|

\---

## 9\. Results, Analysis, and Recommendations

Full results tables, the confusion matrix, modality-contribution analysis, class-reduction impact, and the
final recommended configuration are documented separately in
[**`docs/RESULTS\\\_EXPLAINED.md`**](RESULTS_EXPLAINED.md) — written as a standalone, plain-language walkthrough
of every experimental output.

**Headline takeaways:**

* Removing the ambiguous **Hateful** class improved cross-validated accuracy for every model, by roughly
9–19 percentage points — it overlaps significantly with Toxic and Violence, making it hard to separate.
* **Video features are the single strongest modality** (67.8% accuracy alone), ahead of text/BERT (59.3%) and
audio (58.1%).
* Combining modalities compounds performance: Video → +BERT → +Audio gives consistent, if diminishing, gains.
* **XGBoost** is the strongest generalising model in cross-validation on the 4-class task (84.72% accuracy,
84.46% F1); a **Neural Network** achieves the highest single-split accuracy (89.47%) but shows signs of
overfitting at this dataset size (661 samples).
* **Logistic Regression** is a surprisingly strong, interpretable baseline — 84.12% cross-validated accuracy
on the 4-class task — making it a practical choice for deployment where interpretability and low compute
cost matter.

\---

## 10\. Limitations

* **Dataset size:** 661 samples is workable for classical ML but limits the potential of higher-capacity
models (evidenced by the Neural Network's single-split vs. cross-validated gap).
* **Class imbalance:** stratification was used throughout, but minority-class performance may still be
affected.
* **Language scope:** the dataset may contain multilingual content; BERT embeddings may vary in effectiveness
across languages.
* **No hyperparameter tuning:** models were run with fixed, reasonable defaults rather than
`GridSearchCV`/`RandomSearchCV`-optimised parameters.
* **No dimensionality reduction:** the \~4,614-dim feature space was used directly, without PCA or feature
selection.

## 11\. Future Work

* Scale the dataset (targeting thousands of samples) to improve generalisation.
* Hyperparameter tuning via `GridSearchCV` / `RandomSearchCV`.
* Dimensionality reduction (PCA or feature selection) on the \~4,614-dim space.
* Class-imbalance handling: SMOTE, class weighting, or cost-sensitive learning.
* Explainability: SHAP or LIME for model interpretability.
* Active learning for efficient labelling of ambiguous cases.
* Ensemble methods: stacking or voting classifiers for further performance gains.

\---

## References

* Devlin, J., Chang, M.-W., Lee, K., \& Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding.* NAACL-HLT.
* Chen, T., \& Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
* Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning
Research, 12, 2825–2830.
* Liang, H. (2025). *Embedding-based Retrieval in Multi-Modal Content Moderation.* SIGIR '25.
* Aggarwal, S. (2023). *A Survey of Methodologies for Filtering Child Unsafe Content on YouTube.*
* *Recent Advances in Hate Speech Moderation: Multimodality and the Role of Large Models* (2024). arXiv preprint.
* *AllGuard: A Unified Framework for Multimodal Content Security Assessment* (2025).
* *A Comprehensive Survey on Deep Multi-Source Visual Fusion with Transformer Models for Audio and Text
Content Filtering* (2026). ResearchGate.
* *Multimodal Video Understanding: A Capability-Based Survey of Alignment, Expression, and Reasoning* (2026). MDPI.
* *Video Understanding with Large Language Models: A Survey* (2025).

## Abbreviations

|Abbreviation|Full Form|
|-|-|
|BERT|Bidirectional Encoder Representations from Transformers|
|CV|Cross-Validation|
|F1|F1-Score|
|KNN|K-Nearest Neighbors|
|MFCC|Mel-Frequency Cepstral Coefficients|
|MLP|Multi-Layer Perceptron|
|NB|Naive Bayes|
|PCA|Principal Component Analysis|
|RF|Random Forest|
|SHAP|SHapley Additive exPlanations|
|SMOTE|Synthetic Minority Over-sampling Technique|
|SVM|Support Vector Machine|
|XGBoost|Extreme Gradient Boosting|



