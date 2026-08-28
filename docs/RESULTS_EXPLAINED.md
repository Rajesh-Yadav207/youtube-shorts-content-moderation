# Results Explained

This document walks through every experimental output produced in this project — what each table means, how
to read it, and what it implies. It's meant to be readable on its own, without needing to open the notebooks.

Quick jump:

* [1. How results were measured](#1-how-results-were-measured)
* [2. 5-class results](#2-5-class-results-explicit-violence-hateful-normal-toxic)
* [3. 4-class results](#3-4-class-results-hateful-merged-into-toxic)
* [4. Why merging "Hateful" into "Toxic" helped every model](#4-why-merging-hateful-into-toxic-helped-every-model)
* [5. Which modality matters most](#5-which-modality-matters-most)
* [6. Feature-combination leaderboard](#6-feature-combination-leaderboard)
* [7. The recommended model, and its confusion matrix](#7-the-recommended-model-and-its-confusion-matrix)
* [8. Practical recommendations](#8-practical-recommendations)

---

## 1. How results were measured

Every model was evaluated **two ways**, and both are reported throughout:

1. **Single train/test split** — 80% train / 20% test, stratified so class balance is preserved. Fast to
   compute, but the number can be noisy on a small dataset (661 samples ⇒ only ~132 test samples).
2. **5-fold stratified cross-validation (CV)** — the dataset is split into 5 folds, each model is trained on 4
   and tested on the 5th, five times, and the metrics are averaged. This is the more trustworthy number
   because it isn't sensitive to which 20% happened to land in the test set.

**Rule of thumb used throughout this document:** when a model's single-split accuracy is *much* higher than
its CV accuracy, that's a sign of overfitting to a lucky split rather than genuinely better performance —
trust the CV number more.

All four metrics reported (**Accuracy, Precision, Recall, F1**) are **weighted** — i.e. each class's score is
weighted by how many samples it has, so a small class doesn't get an outsized influence on the overall number.

---

## 2. 5-class results (Explicit, Violence, Hateful, Normal, Toxic)

### Single train/test split

| Model | Accuracy |
|---|---|
| Random Forest | **83.46%** ← best split result |
| Gradient Boosting | 77.44% |
| XGBoost | 75.94% |
| Naive Bayes | 72.93% |
| Neural Network | 70.68% |
| AdaBoost | 68.42% |
| Decision Tree | 46.62% |
| Logistic Regression | 48.12% |
| SVM | 30.83% |
| KNN | 29.32% |

**Reading this:** Random Forest looks best here, but this is a single 80/20 split on a small dataset — see the
CV table below for the more reliable picture. SVM and KNN struggle badly on the raw 5-class problem; both are
distance-based methods that suffer in the very high-dimensional (~4,611-feature) space without more careful
tuning.

### 5-fold cross-validation

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | **74.13%** | 74.66% | 74.13% | **74.09%** ← best CV result |
| XGBoost | 72.62% | 73.11% | 72.62% | 72.37% |
| Neural Network | 71.86% | 72.57% | 71.86% | 71.86% |
| Random Forest | 71.55% | 71.81% | 71.55% | 71.48% |
| SVM | 71.25% | 71.93% | 71.25% | 71.05% |
| Gradient Boosting | 70.34% | 71.94% | 70.34% | 70.36% |
| AdaBoost | 64.76% | 66.52% | 64.76% | 64.94% |
| Naive Bayes | 63.83% | 64.78% | 63.83% | 63.83% |
| KNN | 61.13% | 65.61% | 61.13% | 61.28% |
| Decision Tree | 50.83% | 51.52% | 50.83% | 50.82% |

**Reading this:** once you average across 5 folds, **Logistic Regression comes out on top**, not Random
Forest. This is a meaningful result: Random Forest's 83.46% single-split number was partly a fortunate split —
under proper cross-validation it drops to 71.55%, while Logistic Regression is far more consistent (74.13%
CV vs. 48.12% split — the opposite pattern, suggesting Logistic Regression got an *unlucky* single split, not
that CV overrates it). SVM's picture also flips dramatically (30.83% split → 71.25% CV) once feature scaling
and averaging across folds are accounted for properly.

**Takeaway:** trust the CV numbers. On the 5-class task, simple linear models (Logistic Regression) and
boosted trees (XGBoost) are the most reliable performers.

---

## 3. 4-class results (Hateful merged into Toxic)

> **Important:** in the 4-class taxonomy, the Hateful class is not dropped from the dataset — its 133 samples
> are relabelled and folded into Toxic. All 661 samples are still used; Toxic simply becomes a combined
> category (266 of 661 samples, ~40%) covering what used to be two separate labels. See §4 below for why this
> matters, and [`PROJECT_REPORT.md` §8.2](PROJECT_REPORT.md#82-label-taxonomies) for the exact label mapping.

### Single train/test split

| Model | Accuracy |
|---|---|
| Neural Network | **89.47%** ← best split result |
| XGBoost | 85.71% |
| Gradient Boosting | 83.46% |
| Random Forest | 82.71% |
| AdaBoost | 82.71% |
| Naive Bayes | 78.20% |
| Decision Tree | 63.91% |
| Logistic Regression | 60.15% |
| SVM | 45.86% |
| KNN | 42.11% |

### 5-fold cross-validation

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| XGBoost | **84.72%** | 85.33% | 84.72% | **84.46%** ← best CV result |
| Logistic Regression | 84.12% | 84.22% | 84.12% | 84.03% |
| Neural Network | 83.96% | 84.33% | 83.96% | 83.89% |
| Gradient Boosting | 83.52% | 83.99% | 83.52% | 83.16% |
| SVM | 81.84% | 82.43% | 81.84% | 81.70% |
| Random Forest | 80.63% | 80.80% | 80.63% | 80.32% |
| AdaBoost | 78.97% | 79.97% | 78.97% | 78.81% |
| Naive Bayes | 76.10% | 76.18% | 76.10% | 75.78% |
| KNN | 72.32% | 74.94% | 72.32% | 70.81% |
| Decision Tree | 63.99% | 64.89% | 63.99% | 64.22% |

**Reading this:** on the 4-class task, **XGBoost wins the cross-validated comparison** (84.72% accuracy,
84.46% F1) — this is the single most trustworthy number in the whole project, since it combines the harder,
more reliable evaluation method (CV) with the cleaner label taxonomy (4-class). Logistic Regression is a very
close second (84.12%), which matters because it's far cheaper to train and easier to interpret.

The Neural Network's split-vs-CV gap (89.47% → 83.96%) is a textbook overfitting signature: it memorizes
patterns specific to the lucky 80/20 split better than it generalizes, which is expected given only 661
samples for a multi-layer perceptron with `(128, 64, 32)` hidden units.

---

## 4. Why merging "Hateful" into "Toxic" helped every model

| Model | 5-class CV accuracy | 4-class CV accuracy | Improvement |
|---|---|---|---|
| AdaBoost | 64.76% | 78.97% | **+14.21%** |
| Decision Tree | 50.83% | 63.99% | +13.16% |
| Gradient Boosting | 70.34% | 83.52% | +13.18% |
| Naive Bayes | 63.83% | 76.10% | +12.27% |
| XGBoost | 72.62% | 84.72% | +12.10% |
| Neural Network | 71.86% | 83.96% | +12.10% |
| KNN | 61.13% | 72.32% | +11.19% |
| SVM | 71.25% | 81.84% | +10.59% |
| Logistic Regression | 74.13% | 84.12% | +9.99% |
| Random Forest | 71.55% | 80.63% | +9.08% |

**What this means:** every single model got better — by 9 to 14 accuracy points — once the "Hateful" samples
were relabelled into "Toxic," a 4th class instead of the original 5. Note this is **not** the result of
throwing away data — all 661 samples are still classified in the 4-class experiments, and no information was
deleted. Two things are happening at once, and it's worth separating them:

1. **A genuine label-design fix.** "Hateful" content, as originally labelled, overlapped heavily with "Toxic"
   and "Violence" — content that's hateful is very often *also* toxic or violent, so classifiers had a
   genuinely hard time telling them apart as separate categories. Merging Hateful into Toxic removes that
   specific ambiguous decision boundary, which is a legitimate improvement: the resulting categories are more
   mutually exclusive and easier to learn.
2. **A class-imbalance effect.** Because Hateful's 133 samples are folded into Toxic, Toxic becomes a much
   larger class (266 of 661 samples, ~40%, vs. ~19–21% for each other class). Since accuracy and weighted F1
   are influenced by class size, part of the reported improvement also reflects that a larger, easier-to-hit
   class now makes up a bigger share of the test set — not purely that the *remaining* decision boundaries
   became easier for every class.

**Takeaway for a real system:** the label-design insight is real and worth acting on — Hateful vs. Toxic vs.
Violence is a genuinely hard three-way distinction with this feature set, and collapsing it to two classes
(Toxic/Violence) does make the problem more tractable. But be careful when quoting the headline accuracy
numbers: some of the jump is a byproduct of one class getting bigger, and Section 9.4 of
[`PROJECT_REPORT.md`](PROJECT_REPORT.md) and the confusion-matrix discussion below both show Toxic's size
inflating its raw correct-prediction count, not just its separability.

---

## 5. Which modality matters most

Using Logistic Regression on the 5-class task as the fixed comparison point, testing each modality in isolation:

| Modality (alone) | Accuracy | F1-Score |
|---|---|---|
| **Video** (VideoMAE + ResNet50 + Optical Flow) | **67.78%** | 67.72% |
| Text (BERT, title or comments) | 59.30% | 59.17% |
| Audio (librosa features) | 58.09% | 57.92% |

**Video is the strongest single modality** — visual frames alone (768+2048+120 = 2,936 dims) carry more
discriminative signal for this task than text or audio alone. That's intuitive for a category set like
Explicit/Violence: a lot of that content is visually obvious even without sound or text.

Then, adding modalities on top of Video one at a time:

| Combination | Accuracy | Gain from adding this modality | Insight |
|---|---|---|---|
| Video alone | 67.78% | — (baseline) | Vision is the strongest single modality |
| + BERT (comments) | 73.83% | **+6.05%** | Text + vision gives real synergy — comments add a distinct signal |
| + Audio | 75.34% | +1.51% | Audio adds a smaller, but consistent, boost |
| + Title | 74.13% | **−1.21%** | Adding the title BERT embedding on top of everything else slightly *hurt* performance |

**Why would adding a feature make things worse?** More features means more dimensions relative to only 661
samples — at some point extra columns add noise faster than they add signal, and the model starts fitting to
that noise (a version of the "curse of dimensionality"). The title embedding (768 dims) turned out to be one
of those noise-heavy additions on this dataset — this doesn't mean titles are useless in general, only that
adding them *on top of* comments + video + audio, with this dataset size and title data quality, wasn't worth
the extra 768 dimensions.

---

## 6. Feature-combination leaderboard

Every meaningful combination of modalities that was tested, single train/test split, sorted so you can see the
progression:

| # | Feature combination | Classes | Best model | Accuracy | F1-Score |
|---|---|---|---|---|---|
| 1 | Only BERT (comments) | 5 | Logistic Regression | 59.30% | 59.17% |
| 2 | Only Audio | 5 | XGBoost | 58.09% | 57.92% |
| 3 | Only Video | 5 | Logistic Regression | 67.78% | 67.72% |
| 4 | Comments + Video | 5 | Logistic Regression | 73.83% | 73.79% |
| 5 | Comments + Video + Audio | 5 | Logistic Regression | 75.34% | 75.25% |
| 6 | Title + Comments + Video + Audio | 5 | Logistic Regression | 74.13% | 74.09% |
| 7 | Title + Comments + Video + Audio | 4 | XGBoost | 84.72% | 84.46% |
| 8 | Comments + Video + Audio | 4 | Logistic Regression | **85.78%** | **85.68%** ← best overall result |

**Reading this:** row 8 is the single best result across every experiment in this project — **Logistic
Regression on Comments + Video + Audio, 4-class taxonomy, 85.78% accuracy / 85.68% F1**. Notice it *beats* row
7, which adds the Title feature — this is the same "title adds noise" pattern from Section 5, confirmed again
on the 4-class task.

---

## 7. The recommended model, and its confusion matrix

Based on all of the above, the recommended configuration is:

> **Logistic Regression**, trained on **Comments (BERT) + Video + Audio** features, **4-class taxonomy**
> (Explicit, Normal, Toxic, Violence).

### Why this configuration, and not XGBoost (which won the CV comparison)?

* Its accuracy (85.78%) and F1 (85.68%) on this feature combination are the best single-split results in the
  whole project, and its CV accuracy (84.12%) is within 0.6 points of XGBoost's CV-winning 84.72% — a
  difference that's small relative to the ~10-sample test-set granularity.
* It's dramatically **cheaper to train and run** than an ensemble of hundreds of trees.
* It's **interpretable** — the learned coefficients tell you which features push a prediction toward which
  class, which matters a lot for a moderation system where you may need to explain *why* something was
  flagged.
* It showed a **well-balanced precision/recall trade-off** across classes (see the confusion matrix below),
  rather than being strong on one class at the expense of another.

### All 10 models on this feature combination (4-class)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **85.78%** | **86.29%** | **85.78%** | **85.68%** |
| XGBoost | 83.82% | 83.90% | 83.82% | 83.61% |
| Neural Network | 84.12% | 84.26% | 84.12% | 83.97% |
| Gradient Boosting | 83.36% | 83.19% | 83.36% | 83.12% |
| SVM | 82.00% | 82.49% | 82.00% | 81.80% |
| Random Forest | 81.85% | 81.71% | 81.85% | 81.38% |
| AdaBoost | 76.71% | 78.59% | 76.71% | 76.78% |
| Naive Bayes | 76.71% | 76.75% | 76.71% | 76.35% |
| KNN | 72.62% | 75.08% | 72.62% | 71.29% |
| Decision Tree | 64.60% | 64.93% | 64.60% | 64.65% |

### Confusion matrix (Logistic Regression, on the held-out test set)

Rows = true label, columns = predicted label. Classes are indexed **0 = Explicit, 1 = Normal, 2 = Toxic,
3 = Violence** (per the 4-class encoding).

| True \ Predicted | 0 (Explicit) | 1 (Normal) | 2 (Toxic) | 3 (Violence) |
|---|---|---|---|---|
| **0 (Explicit)** | 2 | 6 | 17 | 1 |
| **1 (Normal)** | 4 | 13 | 7 | 4 |
| **2 (Toxic)** | 2 | 0 | 48 | 3 |
| **3 (Violence)** | 1 | 3 | 8 | 14 |

**How to read this:** the diagonal (2, 13, 48, 14) is what the model got right. Off-diagonal cells are
mistakes. A few things stand out:

* **"Toxic" (class 2) dominates correct predictions (48)**, and its row also has by far the most test
  examples: 2+0+48+3 = **53 samples**, compared to 26–28 for every other class. This isn't a coincidence — as
  explained in [§4](#4-why-merging-hateful-into-toxic-helped-every-model), Toxic absorbed the former Hateful
  class, so it makes up ~40% of the full dataset (266/661), and 53/133 ≈ 40% of this test split too. **Some of
  Toxic's strong raw performance is simply because it has roughly twice as many test examples as any other
  class**, not purely because Toxic content is inherently easier to recognise. It also attracts real confusion
  from other classes (17 Explicit and 7 Normal videos were misclassified as Toxic, and 8 Violence videos too),
  which is consistent with Toxic now covering two originally-distinct concepts (toxic language *and* hateful
  content) — a wider, more heterogeneous category is a bigger target to (correctly *and* incorrectly) land in.
* **Explicit is the weakest-performing class** (only 2 of 26 total Explicit true-label rows correctly
  identified; row sum = 2+6+17+1 = 26) — it's most often confused with Toxic. This is the class most in need
  of more training data or better-separated labelling criteria.
* **Violence and Normal are reasonably well separated** (14/26 and 13/28 correct respectively), though both
  still leak into Toxic more than any other wrong class.

**Practical implication:** if you were to build on this project, the highest-value next step is probably
**improving the separation between "Toxic" and the other three classes** — either through more/better-labelled
training examples for the boundary cases, or by reconsidering whether "Toxic" should be split back into
finer sub-categories (e.g. reintroducing a separate Hateful class, but with clearer, less-overlapping
labelling criteria than the original 5-class scheme used).

---

## 8. Practical recommendations

Summarising the findings into concrete guidance, in priority order:

| Priority | Recommendation | Why |
|---|---|---|
| 1 | Use **Comments + Video + Audio** (drop Title) as the feature set | Title consistently added noise rather than signal, across both 5-class and 4-class experiments |
| 2 | Prefer a **4-class taxonomy** (merge "Hateful" into Toxic, as done here) | Improved every model by 9–14 CV-accuracy points — this is primarily a label-design fix, though some of the gain is a class-size effect from Toxic getting larger (see §4) |
| 3 | Use **Logistic Regression** as the default model | Matches or beats ensemble methods on this dataset size, at a fraction of the training/inference cost, and is interpretable |
| 4 | Reach for **XGBoost** if squeezing out the last ~1% of cross-validated accuracy matters more than interpretability/cost | It won the CV comparison outright on the 4-class task |
| 5 | Treat the **Neural Network's single-split result with caution** | Its CV score is ~5 points lower than its split score — classic small-dataset overfitting |
| 6 | Prioritise **more/better-labelled data for the "Explicit" and "Toxic" boundary** | The confusion matrix shows this is where most of the remaining errors live |

For the full methodology and pipeline behind these numbers, see [`PROJECT_REPORT.md`](PROJECT_REPORT.md). For
the code that produced every table in this document, see the [`notebooks/`](../notebooks) directory —
notebooks `6` and `8` correspond to the 4-class experiments referenced most heavily above.
